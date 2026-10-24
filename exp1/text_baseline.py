"""PLAN 1단계 본 관문의 자연어 쪽 — 같은 전달량(토큰 k개)에서 자연어 메시지로 수신자가 얼마나 맞히는가.

잠재 채널의 벡터 k개와 같은 수의 토큰 k개를 수신자 앞에 텍스트로 넣고, 처음 보는 문서 200개에서 정답률을 잰다.

    oracle  : 사람이 정한 가장 짧은 형식("93V8=5781 YPJU=5659 ...")과 문서 원문을 k토큰에서 자른 것. 학습 없음
    prompt  : 송신자에게 "가능한 한 짧게 다시 쓰라"고 시켜 k토큰까지 생성. 학습 없음
    lora    : 송신자에 LoRA를 붙여, 수신자가 많이 맞힌 요약을 골라 모방 학습하는 것을 여러 번 반복

    python3 exp1/text_baseline.py --mode oracle --k-list 4,8,16,32,64
    python3 exp1/text_baseline.py --mode lora --k-list 16 --tag lora_k16
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_overfit import exact_match, load_frozen, make_docs, pick_device, qa_prompt, receiver_targets  # noqa: E402
from stage1_depth import add_sender_lora  # noqa: E402

INSTRUCTION = ("Rewrite the following as compactly as possible. Keep every code and its value exactly; "
               "drop everything else.\n\n{doc}")


def truncate(tok, text, k):
    ids = tok(text, add_special_tokens=False).input_ids[:k]
    return tok.decode(ids)


def compact(doc):
    return " ".join(f"{key}={val}" for key, val in doc.pairs)


@torch.no_grad()
def score_messages(tok, model, docs, messages, device, batch=32):
    """메시지마다 그 문서의 질문 전부를 묻고, (문서별 맞힌 개수 리스트, 전체 정답률)을 돌려준다."""
    items = [(i, q) for i, d in enumerate(docs) for q in range(len(d.pairs))]
    hits = torch.zeros(len(docs))
    for s in range(0, len(items), batch):
        chunk = items[s: s + batch]
        prompts = [messages[i] + qa_prompt(docs[i], q) for i, q in chunk]
        targets = [" " + docs[i].answer(q) for i, q in chunk]
        em = exact_match(*receiver_targets(tok, model, None, prompts, targets, device)).cpu()
        for (i, _), v in zip(chunk, em.tolist()):
            hits[i] += v
    n_q = sum(len(d.pairs) for d in docs)
    return hits.tolist(), hits.sum().item() / n_q


def chat_inputs(tok, docs, device):
    texts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION.format(doc=d.text)}],
                                     tokenize=False, add_generation_prompt=True) for d in docs]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc.input_ids.to(device), enc.attention_mask.to(device), texts


@torch.no_grad()
def generate(tok, model, docs, k, device, gate=None, sample=False, n=1, batch=32):
    """송신자로 k토큰 이하의 요약을 만든다. sample=True면 문서마다 n개를 뽑는다."""
    out = [[] for _ in docs]
    if gate is not None:
        gate["on"] = True
    for s in range(0, len(docs), batch):
        chunk = docs[s: s + batch]
        ids, mask, _ = chat_inputs(tok, chunk, device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=k, do_sample=sample,
                             temperature=1.0 if sample else None, top_p=0.95 if sample else None, top_k=None,
                             num_return_sequences=n, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, ids.size(1):], skip_special_tokens=True)
        for j, t in enumerate(texts):
            out[s + j // n].append(" " + truncate(tok, t.strip(), k))
    if gate is not None:
        gate["on"] = False
    return out


def sft(tok, model, gate, params, docs, summaries, device, lr, epochs=1, batch=8, gen=None):
    """채팅 프롬프트 -> 요약 토큰에 교차 엔트로피. LoRA만 학습."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    order = list(range(len(docs)))
    gate["on"] = True
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(order), generator=gen).tolist()
        for s in range(0, len(perm), batch):
            idx = [order[i] for i in perm[s: s + batch]]
            _, _, prompts = chat_inputs(tok, [docs[i] for i in idx], device)
            seqs, labels = [], []
            for p, i in zip(prompts, idx):
                p_ids = tok(p, add_special_tokens=False).input_ids
                t_ids = tok(summaries[i].strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
                seqs.append(p_ids + t_ids)
                labels.append([-100] * len(p_ids) + t_ids)
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), tok.pad_token_id)
            lab = torch.full((len(seqs), width), -100)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for j, (x, y) in enumerate(zip(seqs, labels)):
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, :len(y)] = torch.tensor(y)
                att[j, :len(x)] = 1
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1).to(device), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
    gate["on"] = False
    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "prompt", "lora"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k-list", default="4,8,16,32,64")
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=800, help="lora: 요약을 뽑아 학습할 학습 문서 수")
    ap.add_argument("--samples", type=int, default=8, help="lora: 문서당 뽑는 요약 수")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tok, model = load_frozen(args.model, device)
    heldout = make_docs(args.n_heldout, seed=args.data_seed + 10_000)
    ks = [int(x) for x in args.k_list.split(",")]
    bits_per_token = math.log2(len(tok))
    result = {"args": vars(args), "bits_per_token": bits_per_token, "by_k": {}}
    t0 = time.time()
    print(f"mode={args.mode} device={device} heldout={len(heldout)} k={ks} bits/token={bits_per_token:.1f}", flush=True)

    for k in ks:
        row = {}
        if args.mode == "oracle":
            for name, fn in (("compact", compact), ("doc", lambda d: d.text)):
                msgs = [" " + truncate(tok, fn(d), k) for d in heldout]
                _, acc = score_messages(tok, model, heldout, msgs, device)
                row[name] = acc
                row[name + "_example"] = msgs[0]
            # 하한: 메시지 없이
            _, row["none"] = score_messages(tok, model, heldout, [""] * len(heldout), device)
        elif args.mode == "prompt":
            msgs = [m[0] for m in generate(tok, model, heldout, k, device)]
            _, row["prompt"] = score_messages(tok, model, heldout, msgs, device)
            row["prompt_example"] = msgs[0]
        else:
            gate = {"on": False}
            params = add_sender_lora(model, args.lora_rank, gate)
            train = make_docs(args.n_train, seed=args.data_seed)
            gen = torch.Generator().manual_seed(args.seed)
            msgs = [m[0] for m in generate(tok, model, heldout, k, device, gate)]
            _, acc = score_messages(tok, model, heldout, msgs, device)
            row["rounds"] = [{"round": 0, "heldout": acc, "example": msgs[0]}]
            print(f"k={k} round 0 heldout {acc:.3f}  예: {msgs[0]!r}  {time.time()-t0:.0f}s", flush=True)
            for r in range(1, args.rounds + 1):
                samples = generate(tok, model, train, k, device, gate, sample=True, n=args.samples)
                flat_docs = [d for d, ms in zip(train, samples) for _ in ms]
                flat_msgs = [m for ms in samples for m in ms]
                hits, sample_acc = score_messages(tok, model, flat_docs, flat_msgs, device)
                best_docs, best_msgs = [], []
                for i, d in enumerate(train):
                    h = hits[i * args.samples: (i + 1) * args.samples]
                    j = max(range(len(h)), key=lambda x: h[x])
                    if h[j] > 0:
                        best_docs.append(d)
                        best_msgs.append(samples[i][j])
                loss = sft(tok, model, gate, params, best_docs, best_msgs, device, args.lr, gen=gen)
                msgs = [m[0] for m in generate(tok, model, heldout, k, device, gate)]
                _, acc = score_messages(tok, model, heldout, msgs, device)
                row["rounds"].append({"round": r, "sample_acc": sample_acc, "kept": len(best_docs), "sft_loss": loss,
                                      "heldout": acc, "example": msgs[0]})
                print(f"k={k} round {r} 표본 정답률 {sample_acc:.3f} 채택 {len(best_docs)} sft {loss:.3f} "
                      f"heldout {acc:.3f}  예: {msgs[0]!r}  {time.time()-t0:.0f}s", flush=True)
            row["lora"] = row["rounds"][-1]["heldout"]
            model = load_frozen(args.model, device)[1]  # 다음 k는 LoRA 없는 새 모델에서
        row["bits"] = k * bits_per_token
        result["by_k"][k] = row
        print(f"k={k} " + "  ".join(f"{n} {v:.3f}" for n, v in row.items() if isinstance(v, float)), flush=True)

    out = ROOT / "results" / f"text_{args.tag or args.mode}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
