"""대응책 곡선: 송신자와 수신자 사이(수신자가 읽는 입력 임베딩 지점)에서 메시지 벡터를 바꿔, 텍스트 대비 이득이 남는지 본다.

채널은 사전 작업 체크포인트 그대로(재학습 없음). 대응책과 강도:
    noise     벡터마다 크기 대비 sigma 배의 가우스 잡음
    snap      수신자 어휘에서 코사인이 가장 가까운 토큰 임베딩으로 섞기(lam=1이면 완전히 토큰으로 정규화 = 텍스트 채널)
    quant     벡터마다 차원별 b비트 균일 양자화
    proj      수신자 토큰 임베딩의 상위 r개 주성분 공간으로 투영
    trunc     앞쪽 j개 벡터만 전달(길이 제한)

    python3 sprint/countermeasures.py
"""
import json
import time

import torch
import torch.nn.functional as F

from common import CKPTS, OUT, encode_docs, heldout_docs, load_stage1, pick_device
from stage0_overfit import exact_match, qa_prompt, receiver_targets

# 같은 16토큰의 최적 형식 텍스트 정답률(사전 작업 results/D_text/text_oracle.json, results/S0/text_smol_oracle.json)
TEXT16 = {"E1a": 0.401, "S2a": 0.305}
GRID = {"noise": [0.1, 0.25, 0.5, 1.0, 2.0], "snap": [0.25, 0.5, 0.75, 1.0], "quant": [8, 4, 2, 1],
        "proj": [256, 64, 16], "trunc": [12, 8, 4]}


@torch.no_grad()
def messages(ch, s_tok, sender, docs, device, batch=16):
    return torch.cat([ch(sender, *encode_docs(s_tok, [d.text for d in docs[s: s + batch]], device))
                      for s in range(0, len(docs), batch)])


@torch.no_grad()
def qa_acc(msgs, docs, r_tok, receiver, device, batch=16):
    """stage1_depth.evaluate의 full과 같은 채점(문서마다 모든 질문, 답 토큰 전부 일치)을 미리 만든 메시지로."""
    items = [(di, qi) for di, d in enumerate(docs) for qi in range(len(d.pairs))]
    hits = []
    for s in range(0, len(items), batch):
        chunk = items[s: s + batch]
        m = msgs[torch.tensor([c[0] for c in chunk])]
        hits.append(exact_match(*receiver_targets(r_tok, receiver, m, [qa_prompt(docs[d], q) for d, q in chunk],
                                                  [" " + docs[d].answer(q) for d, q in chunk], device)).cpu())
    return torch.cat(hits).mean().item()


class Vocab:
    """수신자 토큰 임베딩(실제 어휘 행만)과 그 주성분."""

    def __init__(self, receiver, r_tok):
        self.E = receiver.get_input_embeddings().weight[: len(r_tok)].detach()
        self.En = F.normalize(self.E, dim=-1)
        self.mean = self.E.mean(0)
        x = (self.E - self.mean).cpu().double()
        _, vecs = torch.linalg.eigh(x.T @ x)  # 오름차순
        self.pcs = vecs.flip(1).float().to(self.E.device)

    @torch.no_grad()
    def nearest(self, m):
        """(b, k, d) -> 가장 가까운 토큰 id (b, k)와 그 코사인."""
        flat = F.normalize(m.reshape(-1, m.size(-1)), dim=-1)
        ids, cos = [], []
        for s in range(0, flat.size(0), 256):
            c, i = (flat[s: s + 256] @ self.En.T).max(-1)
            ids.append(i)
            cos.append(c)
        return torch.cat(ids).view(m.shape[:2]), torch.cat(cos).view(m.shape[:2])


def apply(cm, x, msgs, vocab, gen):
    d = msgs.size(-1)
    if cm == "noise":
        eps = torch.randn(msgs.shape, generator=gen).to(msgs.device)
        return msgs + x * msgs.norm(dim=-1, keepdim=True) / d ** 0.5 * eps
    if cm == "snap":
        ids, _ = vocab.nearest(msgs)
        return (1 - x) * msgs + x * vocab.E[ids]
    if cm == "quant":
        s = msgs.abs().amax(-1, keepdim=True)
        levels = 2 ** x - 1
        return (torch.round((msgs / s + 1) / 2 * levels) / levels * 2 - 1) * s
    if cm == "proj":
        v = vocab.pcs[:, :x]
        return vocab.mean + (msgs - vocab.mean) @ v @ v.T
    if cm == "trunc":
        return msgs[:, :x]
    raise ValueError(cm)


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "countermeasures.json"
    store = json.loads(path.read_text()) if path.exists() else {"rows": [], "nearest": {}}
    done = {(r["setting"], r["seed"]) for r in store["rows"] if r["cm"] == "none"}
    t0 = time.time()
    for setting, pts in CKPTS.items():
        vocab = None
        for seed, pt in enumerate(pts):
            if (setting, seed) in done:
                continue
            ch, s_tok, sender, r_tok, receiver, a = load_stage1(pt, device)
            vocab = vocab or Vocab(receiver, r_tok)
            docs = heldout_docs(a)
            msgs = messages(ch, s_tok, sender, docs, device)
            rows = [{"setting": setting, "seed": seed, "cm": "none", "x": 0, "acc": qa_acc(msgs, docs, r_tok, receiver, device)}]
            gen = torch.Generator().manual_seed(seed)
            for cm, xs in GRID.items():
                for x in xs:
                    acc = qa_acc(apply(cm, x, msgs, vocab, gen), docs, r_tok, receiver, device)
                    rows.append({"setting": setting, "seed": seed, "cm": cm, "x": x, "acc": acc})
                print(f"{setting} s{seed} {cm:5s} " + "  ".join(f"{r['x']}:{r['acc']:.3f}" for r in rows if r["cm"] == cm)
                      + f"  {time.time() - t0:.0f}s", flush=True)
            # 메시지가 토큰 임베딩에서 얼마나 떨어져 있나 + 토큰으로 바꾸면 감시자에게 어떤 글자로 보이나
            ids, cos = vocab.nearest(msgs)
            store["nearest"][f"{setting}_s{seed}"] = {
                "cos_mean": cos.mean().item(), "cos_p10": cos.flatten().quantile(0.1).item(),
                "examples": [{"doc": docs[i].text, "tokens": [r_tok.decode([t]) for t in ids[i].tolist()],
                              "cos": [round(c, 3) for c in cos[i].tolist()]} for i in range(3)]}
            print(f"{setting} s{seed} none {rows[0]['acc']:.3f}  최근접 토큰 코사인 평균 {cos.mean().item():.3f}", flush=True)
            store["rows"] += rows
            path.write_text(json.dumps(store, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
