"""공통 Z 채널(스프린트 중 다시 돌린 2-3·2-4 체크포인트)에도 4.1의 대응책이 통하는지 본다.

방향마다(2-3은 Qwen·SmolLM2 4방향, 2-4는 OLMo까지 9방향) 같은 대응책을 걸고, 수신 모델별 16토큰 텍스트 대조도 함께 잰다.
투영은 텍스트까지 지워 선택성이 없으므로(4.1) 뺐다. 보류 문서는 stage2_shared.py와 같다(겹침 제거 없음).

    python3 sprint/countermeasures_shared.py
"""
import json
import time

import torch

from common import OUT, ROOT, make_docs, model, pick_device, receiver_tok
from countermeasures import Vocab, apply, messages, qa_acc
from stage1_depth import DeepChannel
from stage2_shared import Pair
from text_baseline import compact

GRID = {"snap": [0.25, 1.0], "noise": [0.25, 0.5], "quant": [4, 2], "trunc": [8]}
CKPTS = [("S23r", s, ROOT / f"results/stage2_S23r_s{s}.pt") for s in range(3)] + \
        [("S24r", s, ROOT / f"results/stage2_S24r_s{s}.pt") for s in range(3)]


def short(name):
    return name.split("/")[-1].split("-")[0]


def load_shared(pt, device):
    """stage2_shared.py가 저장한 체크포인트 -> 모델 이름 목록, 모델별 채널(인코더+디코더)."""
    ck = torch.load(pt, map_location="cpu")
    chans = {}
    for name, layer in zip(ck["models"], ck["layers"]):
        m = model(name, device)[1]
        sd = ck["channels"][name.replace("/", "__").replace(".", "_")]
        ch = DeepChannel(m, m, sd["queries"].shape[0], sd["queries"].shape[1], [layer])
        ch.load_state_dict(sd)
        chans[name] = ch.to(device).eval().requires_grad_(False)
    return ck["models"], chans


def sweep(msgs, docs, r_tok, receiver, vocab, gen, device):
    out = [("none", 0, qa_acc(msgs, docs, r_tok, receiver, device))]
    for cm, xs in GRID.items():
        out += [(cm, x, qa_acc(apply(cm, x, msgs, vocab, gen), docs, r_tok, receiver, device)) for x in xs]
    return out


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "countermeasures_shared.json"
    docs = make_docs(200, seed=10_000)
    rows, vocabs, toks, t0 = [], {}, {}, time.time()
    for tag, seed, pt in CKPTS:
        names, chans = load_shared(pt, device)
        for r in names:
            if r not in vocabs:
                toks[r] = receiver_tok(r)
                vocabs[r] = Vocab(model(r, device)[1], toks[r])
                ids = torch.tensor([toks[r](" " + compact(d), add_special_tokens=False).input_ids[:16] for d in docs],
                                   device=device)
                gen = torch.Generator().manual_seed(0)
                for cm, x, acc in sweep(vocabs[r].E[ids], docs, toks[r], model(r, device)[1], vocabs[r], gen, device):
                    rows.append({"ckpt": "text", "seed": 0, "dir": f"text>{short(r)}", "cm": cm, "x": x, "acc": acc})
                print(f"text>{short(r)}  " + "  ".join(f"{q['cm']}{q['x']}:{q['acc']:.3f}" for q in rows if q["dir"] == f"text>{short(r)}"),
                      flush=True)
        for s in names:
            s_tok, sender = model(s, device)
            for r in names:
                msgs = messages(Pair(chans[s], chans[r]), s_tok, sender, docs, device)
                gen = torch.Generator().manual_seed(seed)
                res = sweep(msgs, docs, toks[r], model(r, device)[1], vocabs[r], gen, device)
                rows += [{"ckpt": tag, "seed": seed, "dir": f"{short(s)}>{short(r)}", "cm": cm, "x": x, "acc": acc} for cm, x, acc in res]
                print(f"{tag} s{seed} {short(s)}>{short(r)}  " + "  ".join(f"{cm}{x}:{acc:.3f}" for cm, x, acc in res)
                      + f"  {time.time() - t0:.0f}s", flush=True)
        path.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
