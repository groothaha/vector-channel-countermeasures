"""PLAN 2단계 2-3·2-4 — 여러 모델이 공유하는 공간 Z.

모델마다 채널을 하나씩 둔다(stage1_depth.DeepChannel: 인코더 = 풀러, 디코더 = MLP). 방향 s→r의 메시지는
"s의 인코더 → Z → r의 디코더"다. 같은 Z를 모든 방향이 거치므로 Z는 특정 쌍 전용이 아니라 공용 공간이 된다.

    2-3 공동 학습:  python3 exp1/stage2_shared.py --tag j0 --models Q,S --layers 3,4
    2-4 모델 부착:  python3 exp1/stage2_shared.py --tag a0 --load results/stage2_j0.pt --add O --add-layer 2

부착 모드에서는 불러온 채널을 전부 동결하고, 새 모델의 인코더·디코더만 새 모델이 끼는 방향으로 학습한다.
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_overfit import (RECON_PROMPT, align_loss, ce_loss, encode_docs, load_frozen, make_docs, pick_device,  # noqa: E402
                            qa_prompt, receiver_targets)
from stage1_depth import DeepChannel, evaluate  # noqa: E402

ALIASES = {"Q": "Qwen/Qwen2.5-0.5B-Instruct", "S": "HuggingFaceTB/SmolLM2-360M-Instruct",
           "O": "allenai/OLMo-2-0425-1B-Instruct"}


class Pair(nn.Module):
    """방향 하나: 송신 모델의 인코더 + 수신 모델의 디코더. evaluate()가 채널처럼 부를 수 있게 감싼다."""

    def __init__(self, enc, dec):
        super().__init__()
        self.enc, self.dec = enc, dec

    def forward(self, sender, doc_ids, doc_mask):
        return self.dec.to_receiver(self.enc.to_z(self.enc.read(sender, doc_ids, doc_mask), doc_mask))


def align_for(enc, dec, receiver, r_ids, r_mask):
    """정렬 예열 목표. 슬롯 수는 메시지를 만드는 인코더의 k를 따른다(부착 모드에서 송신·수신 k가 다를 수 있음)."""
    saved = dec.k
    dec.k = enc.k
    try:
        return dec.align_target(receiver, r_ids, r_mask)
    finally:
        dec.k = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", default="Q,S", help="공동 학습할 모델(별칭 Q·S·O 또는 HF 이름), 쉼표 구분")
    ap.add_argument("--layers", default="3,4", help="모델별 읽을 층, --models와 같은 순서")
    ap.add_argument("--load", default=None, help="부착 모드: 2-3에서 저장한 체크포인트")
    ap.add_argument("--add", default=None, help="부착 모드: 새로 붙일 모델")
    ap.add_argument("--add-layer", type=int, default=None)
    ap.add_argument("--k-schedule", default="32:3000,16:9000")
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=1600)
    ap.add_argument("--warmup-batch", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--task-lr", type=float, default=3e-4)
    ap.add_argument("--task-lr-ramp", type=int, default=300)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-docs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args()

    phases = [(int(a), int(b)) for a, b in (x.split(":") for x in args.k_schedule.split(","))]
    total = sum(n for _, n in phases)
    torch.manual_seed(args.seed)
    device = pick_device()

    ckpt = torch.load(args.load, map_location="cpu") if args.load else None
    if ckpt:
        names, layers = ckpt["models"], ckpt["layers"]
    else:
        names = [ALIASES.get(m, m) for m in args.models.split(",")]
        layers = [int(x) for x in args.layers.split(",")]
    new = ALIASES.get(args.add, args.add) if args.add else None
    if new:
        names, layers = names + [new], layers + [args.add_layer]

    toks, models, chans = {}, {}, nn.ModuleDict()
    for name, layer in zip(names, layers):
        toks[name], models[name] = load_frozen(name, device)
        chans[name.replace("/", "__").replace(".", "_")] = DeepChannel(models[name], models[name], phases[0][0], args.z_dim, [layer])
    key = {n: n.replace("/", "__").replace(".", "_") for n in names}
    chans.to(device)

    if ckpt:
        for n in ckpt["models"]:
            chans[key[n]].resize(ckpt["channels"][key[n]]["queries"].shape[0])  # 저장 시점의 벡터 수에 맞춘다
            chans[key[n]].load_state_dict(ckpt["channels"][key[n]])
            chans[key[n]].requires_grad_(False)
        trainable = [new]
        dirs = [(s, r) for s, r in product(names, names) if new in (s, r)]
    else:
        trainable = names
        dirs = list(product(names, names))
    params = [p for n in trainable for p in chans[key[n]].parameters()]
    print(f"tag={args.tag} device={device} models={names} layers={layers} trainable={trainable} "
          f"params={sum(p.numel() for p in params)/1e6:.2f}M dirs={len(dirs)}", flush=True)

    train = make_docs(args.n_train, seed=args.data_seed)
    heldout = make_docs(args.n_heldout, seed=args.data_seed + 10_000)
    short = lambda n: n.split("/")[-1].split("-")[0]  # noqa: E731
    gen = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history, t0 = [], time.time()

    def sample(n):
        return [train[i] for i in torch.randint(len(train), (n,), generator=gen).tolist()]

    def message(s, r, docs_):
        enc, dec = chans[key[s]], chans[key[r]]
        ids, mask = encode_docs(toks[s], [d.text for d in docs_], device)
        msg = Pair(enc, dec)(models[s], ids, mask)
        r_ids, r_mask = encode_docs(toks[r], [d.text for d in docs_], device)
        return msg, align_loss(msg, align_for(enc, dec, models[r], r_ids, r_mask))

    def report(step, final=False):
        row = {"step": step, "sec": time.time() - t0}
        docs_ = heldout if final else heldout[: args.eval_docs]
        for s, r in dirs:
            ev = evaluate(Pair(chans[key[s]], chans[key[r]]), toks[s], models[s], toks[r], models[r], docs_, device,
                          args.batch, controls=final)
            row[f"{short(s)}>{short(r)}"] = ev
        history.append(row)
        print(f"step {step:5d}  " + "  ".join(f"{d} {v['full']:.3f}" for d, v in row.items() if isinstance(v, dict))
              + f"  {row['sec']:.0f}s", flush=True)
        return row

    for step in range(1, args.warmup_steps + 1):
        s, r = dirs[step % len(dirs)]
        _, al = message(s, r, sample(args.warmup_batch))
        opt.zero_grad(set_to_none=True)
        al.backward()
        opt.step()
        if step % 200 == 0:
            print(f"warmup {step:4d}  {short(s)}>{short(r)} align {al.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    report(0)

    bounds, acc = {}, 0
    for i, (k_i, n_i) in enumerate(phases):
        bounds[acc + 1] = (i, k_i)
        acc += n_i
    phase_start = 1
    for step in range(1, total + 1):
        if step in bounds and bounds[step][0] > 0:
            for n in trainable:
                chans[key[n]].resize(bounds[step][1])
            params = [p for n in trainable for p in chans[key[n]].parameters()]
            opt = torch.optim.AdamW(params, lr=args.task_lr, weight_decay=0.0)
            phase_start = step
            print(f"--- step {step}: 벡터 수 {bounds[step][1]}로 변경", flush=True)
        frac = min(1.0, (step - phase_start + 1) / max(args.task_lr_ramp, 1))
        for g in opt.param_groups:
            g["lr"] = args.task_lr * (0.1 + 0.9 * frac)
        s, r = dirs[step % len(dirs)]
        bd = sample(args.batch)
        qi = [int(torch.randint(len(d.pairs), (1,), generator=gen)) for d in bd]
        msg, al = message(s, r, bd)
        qa = ce_loss(*receiver_targets(toks[r], models[r], msg, [qa_prompt(d, q) for d, q in zip(bd, qi)],
                                       [" " + d.answer(q) for d, q in zip(bd, qi)], device))
        rec = ce_loss(*receiver_targets(toks[r], models[r], msg, [RECON_PROMPT] * len(bd), [" " + d.text for d in bd], device))
        loss = qa + args.recon_weight * rec + args.align_weight * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.eval_every == 0 and step != total:
            report(step)

    final = report(total, final=True)
    print("최종(보류 전체 %d문서)  " % len(heldout) + "  ".join(
        f"{d} {v['full']:.3f}(swap {v['swapped']:.3f})" for d, v in final.items() if isinstance(v, dict)), flush=True)
    out = ROOT / "results" / f"stage2_{args.tag}"
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "models": names, "layers": layers,
                                                    "trainable": trainable, "history": history}, indent=2, ensure_ascii=False))
    torch.save({"models": names, "layers": layers, "channels": {key[n]: chans[key[n]].state_dict() for n in names}},
               out.with_suffix(".pt"))
    print(f"저장: {out}.json, {out}.pt")


if __name__ == "__main__":
    main()
