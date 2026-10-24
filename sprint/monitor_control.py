"""감시자 판독의 대조 과제(Hewitt & Liang 2019): 학습 문서의 표적을 문서끼리 섞어 붙인 뒤 같은 선형 탐침을 학습한다.

탐침이 메시지 내용이 아니라 학습 표본을 외워서 맞히는 것이라면 이 대조에서도 정답률이 나온다. 우연 수준(자리 0.1)이어야
monitor_probe.py의 결과를 "메시지에서 읽었다"로 해석할 수 있다.

    python3 sprint/monitor_control.py
"""
import json
import time

import torch

from common import CKPTS, OUT, heldout_docs, load_stage1, pick_device, train_docs
from countermeasures import messages
from monitor_probe import WDS, digits, features, fit, score


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "monitor_control.json"
    rows, t0 = [], time.time()
    for setting, pts in CKPTS.items():
        for seed, pt in enumerate(pts):
            ch, s_tok, sender, _, _, a = load_stage1(pt, device)
            tr, ho = train_docs(a), heldout_docs(a)
            m_tr, m_ho = messages(ch, s_tok, sender, tr, device), messages(ch, s_tok, sender, ho, device)
            y_tr, y_ho = digits(tr).to(device), digits(ho).to(device)
            mu, sd = m_tr.mean((0, 1)), m_tr.std((0, 1)) + 1e-6
            x_tr, x_ho = features("linear", (m_tr - mu) / sd), features("linear", (m_ho - mu) / sd)
            perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)).to(device)
            y_sh = y_tr[perm]
            n_val = 200
            best = max(WDS, key=lambda wd: score(fit("linear", x_tr[:-n_val], y_sh[:-n_val], wd), x_tr[-n_val:], y_sh[-n_val:])["digit"])
            net = fit("linear", x_tr, y_sh, best)
            row = {"setting": setting, "seed": seed, "wd": best, "heldout": score(net, x_ho, y_ho), "train_fit": score(net, x_tr, y_sh)}
            rows.append(row)
            path.write_text(json.dumps(rows, indent=1))
            print(f"{setting} s{seed}  섞은 표적 탐침: 보류 자리 {row['heldout']['digit']:.3f} 값 {row['heldout']['value']:.3f}  "
                  f"(학습 적합 자리 {row['train_fit']['digit']:.3f})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
