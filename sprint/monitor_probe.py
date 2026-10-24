"""감시자 판독: 디코더·수신자 없이 메시지 벡터만 보는 감시자가 문서의 값을 복원할 수 있는가.

감시자 가정(강한 쪽): 학습 문서 2000개의 (메시지, 원문) 쌍을 갖고 있어 탐침을 지도 학습할 수 있다.
표적: 문서 순서대로 값 4개 x 숫자 4자리 = 16자리(각 10지선다, 우연 0.1). 보류 문서로 평가.
    linear   메시지 전체(k x d)를 펼친 선형 탐침
    mean     벡터 k개 평균(d)의 선형 탐침 — 위치 정보를 버린 약한 감시자
    mlp      펼친 메시지의 은닉층 하나짜리 탐침

    python3 sprint/monitor_probe.py
"""
import json
import time

import torch
import torch.nn as nn

from common import CKPTS, OUT, heldout_docs, load_stage1, pick_device, train_docs
from countermeasures import messages

PROBES = ("linear", "mean", "mlp")
WDS = (1e-4, 1e-2, 1e-1)


def digits(docs):
    """(n, 16) 정수 표적. 문서 문장 순서의 값 4개, 각 4자리."""
    return torch.tensor([[int(c) for _, v in d.pairs for c in v] for d in docs])


def features(kind, m):
    return m.mean(1) if kind == "mean" else m.flatten(1)


def fit(kind, x, y, wd, steps=600):
    d = x.size(1)
    net = nn.Linear(d, 160) if kind != "mlp" else nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 160))
    net = net.to(x.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(net(x).view(-1, 16, 10).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def score(net, x, y):
    pred = net(x).view(-1, 16, 10).argmax(-1)
    hit = (pred == y).float()
    return {"digit": hit.mean().item(), "value": hit.view(-1, 4, 4).all(-1).float().mean().item()}


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "monitor_probe.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["setting"], r["seed"]) for r in rows}
    t0 = time.time()
    for setting, pts in CKPTS.items():
        for seed, pt in enumerate(pts):
            if (setting, seed) in done:
                continue
            ch, s_tok, sender, _, _, a = load_stage1(pt, device)
            tr, ho = train_docs(a), heldout_docs(a)
            m_tr, m_ho = messages(ch, s_tok, sender, tr, device), messages(ch, s_tok, sender, ho, device)
            y_tr, y_ho = digits(tr).to(device), digits(ho).to(device)
            # 표준화는 학습 메시지 통계로만
            mu, sd = m_tr.mean((0, 1)), m_tr.std((0, 1)) + 1e-6
            m_tr, m_ho = (m_tr - mu) / sd, (m_ho - mu) / sd
            n_val = 200  # 가중 감쇠는 학습 문서 끝 200개로 고른다(보류 문서는 최종 평가에만)
            row = {"setting": setting, "seed": seed}
            for kind in PROBES:
                x_tr, x_ho = features(kind, m_tr), features(kind, m_ho)
                best = max(WDS, key=lambda wd: score(fit(kind, x_tr[:-n_val], y_tr[:-n_val], wd),
                                                     x_tr[-n_val:], y_tr[-n_val:])["digit"])
                row[kind] = {"wd": best, **score(fit(kind, x_tr, y_tr, best), x_ho, y_ho)}
            rows.append(row)
            path.write_text(json.dumps(rows, indent=1))
            print(f"{setting} s{seed}  " + "  ".join(f"{k} 자리 {row[k]['digit']:.3f} 값 {row[k]['value']:.3f}" for k in PROBES)
                  + f"  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
