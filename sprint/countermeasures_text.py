"""대응책의 선택성: 같은 대응책을 평범한 16토큰 텍스트 메시지에 걸었을 때 정답률이 얼마나 남는가.

텍스트 메시지 = 사전 작업 text_baseline의 최적 형식("93V8=5781 YPJU=5659 ...") 앞 16토큰을 수신자 토큰 임베딩으로 바꾼 것.
벡터 메시지와 같은 자리(프롬프트 앞)에 넣고 같은 채점을 한다. 벡터 채널은 끊고 텍스트는 살리는 대응책이 쓸모 있는 방어.

    python3 sprint/countermeasures_text.py
"""
import json
import time

import torch

from common import CKPTS, OUT, heldout_docs, model, pick_device, receiver_tok
from countermeasures import GRID, Vocab, apply, qa_acc
from text_baseline import compact

K = 16
SEEDS = range(3)  # 텍스트 메시지는 같고 잡음만 시드마다 다르다


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for setting, pts in CKPTS.items():
        a = torch.load(pts[0], map_location="cpu")["args"]
        r_name = a.get("receiver") or a["model"]
        receiver, r_tok = model(r_name, device)[1], receiver_tok(r_name)
        vocab = Vocab(receiver, r_tok)
        docs = heldout_docs(a)
        ids = torch.tensor([r_tok(" " + compact(d), add_special_tokens=False).input_ids[:K] for d in docs], device=device)
        msgs = vocab.E[ids]
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(seed)
            rows.append({"setting": setting, "seed": seed, "cm": "none", "x": 0, "acc": qa_acc(msgs, docs, r_tok, receiver, device)})
            for cm, xs in GRID.items():
                for x in xs:
                    rows.append({"setting": setting, "seed": seed, "cm": cm, "x": x,
                                 "acc": qa_acc(apply(cm, x, msgs, vocab, gen), docs, r_tok, receiver, device)})
                print(f"text {setting} s{seed} {cm:5s} " + "  ".join(f"{r['x']}:{r['acc']:.3f}" for r in rows
                                                                      if r["cm"] == cm and r["seed"] == seed and r["setting"] == setting)
                      + f"  {time.time() - t0:.0f}s", flush=True)
            (OUT / "countermeasures_text.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
