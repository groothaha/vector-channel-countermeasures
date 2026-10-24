"""PLAN "구조 깊이 축" — 채널 구조를 바꿔 가며 처음 보는 문서에서의 정답률을 잰다.

0단계(stage0_overfit.py)와 달리 학습 문서와 보류 문서를 나눈다. 구조 변경은 인자로 하나씩 켠다.

    기준선            python3 exp1/stage1_depth.py --tag base
    c. 풀러 깊이       python3 exp1/stage1_depth.py --tag pool3 --pool-layers 3
    d. 여러 층 읽기     python3 exp1/stage1_depth.py --tag multi --layers 6,12,18
    e. 디코더 슬롯 어텐션 python3 exp1/stage1_depth.py --tag decattn --dec-attn 1
    f. 송신자 LoRA     python3 exp1/stage1_depth.py --tag lora --sender-lora 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_overfit import (RECON_PROMPT, PoolChannel, align_loss, ce_loss, encode_docs, exact_match,  # noqa: E402
                            load_frozen, make_docs, mlp, pick_device, qa_prompt, receiver_targets, sinusoid,
                            token_hits)
from transformers import AutoTokenizer  # noqa: E402


class LoRALinear(nn.Module):
    """동결된 선형층에 저랭크 보정을 더한다. gate["on"]이 꺼져 있으면 원래 층 그대로.

    0단계·1단계는 송신자와 수신자가 같은 모델 객체라, 송신자로 읽을 때만 켜고 수신자로 쓸 때는 끈다.
    """

    def __init__(self, base, r, gate):
        super().__init__()
        self.base, self.gate, self.scaling = base, gate, 2.0
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        y = self.base(x)
        if self.gate["on"]:
            y = y + (x @ self.A.T @ self.B.T) * self.scaling
        return y


def add_sender_lora(model, r, gate):
    """모든 층의 q_proj, v_proj에 LoRA를 붙이고 학습 파라미터 목록을 돌려준다."""
    params = []
    for layer in model.base_model.layers:
        for name in ("q_proj", "v_proj"):
            wrapped = LoRALinear(getattr(layer.self_attn, name), r, gate).to(next(model.parameters()).device)
            setattr(layer.self_attn, name, wrapped)
            params += [wrapped.A, wrapped.B]
    return params


class PoolBlock(nn.Module):
    """(질의 간 어텐션) -> 토큰 어텐션 -> MLP. 질의 간 어텐션은 슬롯끼리 무엇을 담을지 조율하는 통로."""

    def __init__(self, d, heads, hidden, query_attn):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True) if query_attn else None
        self.n0 = nn.LayerNorm(d) if query_attn else None
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, hidden, d)

    def forward(self, q, x, bias):
        if self.sa is not None:
            a, _ = self.sa(q, q, q, need_weights=False)
            q = self.n0(q + a)
        a, _ = self.ca(q, x, x, attn_mask=bias, need_weights=False)
        q = self.n1(q + a)
        return self.n2(q + self.ff(q))


class SlotBlock(nn.Module):
    """디코더 쪽 슬롯 간 어텐션 -> MLP."""

    def __init__(self, d, heads, hidden):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, hidden, d)

    def forward(self, z):
        a, _ = self.sa(z, z, z, need_weights=False)
        z = self.n1(z + a)
        return self.n2(z + self.ff(z))


class Gated(nn.Module):
    """블록 출력을 x + g * (블록(x) - x)로 섞는다. g는 0에서 시작하므로 처음에는 블록이 없는 것과 같다.

    1차 실행에서 슬롯을 섞는 블록을 그대로 넣자 예열 끝 정렬 손실이 기준선의 3배(0.2 대 0.066)였고
    과제 학습이 시작되지 않았다. 기준선과 같은 지점에서 출발시켜 블록의 용량 효과만 재기 위한 장치.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.g = nn.Parameter(torch.zeros(()))

    def forward(self, x, *args):
        return x + self.g * (self.block(x, *args) - x)


class DeepChannel(nn.Module):
    """PoolChannel을 일반화. pool_layers=1, layers=[가운데], dec_attn=0이면 PoolChannel과 같은 구조."""

    def __init__(self, sender, receiver, k, z_dim, layers, pool_layers=1, dec_attn=0, hidden=1024, heads=8, gate=None,
                 zero_gate=False, read_mode="pool"):
        super().__init__()
        d_s, d_r = sender.config.hidden_size, receiver.config.hidden_size
        self.k, self.layers, self.z_dim, self.heads = k, layers, z_dim, heads
        self.gate = gate  # 송신자 LoRA 스위치. None이면 송신자에 기울기를 흘리지 않는다
        self.read_mode = read_mode
        if read_mode == "slot":
            # 0단계에서 실패한 슬롯 방식(문서 뒤에 학습 벡터 k개를 붙여 송신자 통과)을 예열과 함께 재시험
            self.slots = nn.Parameter(torch.randn(k, d_s) * sender.get_input_embeddings().weight.std().item())
            self.slot_ff = mlp(z_dim, hidden, z_dim)
            self.slot_norm = nn.LayerNorm(z_dim)
        self.in_proj = nn.Linear(d_s * len(layers), z_dim)
        self.queries = nn.Parameter(torch.randn(k, z_dim) * 0.02)
        if zero_gate:
            # 첫 블록은 기준선과 같은 블록, 추가 블록은 0 게이트로 감싸 처음에는 꺼진 상태
            self.blocks = nn.ModuleList([PoolBlock(z_dim, heads, hidden, query_attn=False)] +
                                        [Gated(PoolBlock(z_dim, heads, hidden, query_attn=True)) for _ in range(pool_layers - 1)])
            self.dec_blocks = nn.ModuleList(Gated(SlotBlock(z_dim, heads, hidden)) for _ in range(dec_attn))
        else:
            self.blocks = nn.ModuleList(PoolBlock(z_dim, heads, hidden, query_attn=pool_layers > 1) for _ in range(pool_layers))
            self.dec_blocks = nn.ModuleList(SlotBlock(z_dim, heads, hidden) for _ in range(dec_attn))
        self.decoder = mlp(z_dim, hidden, d_r)
        self.scale = nn.Parameter(receiver.get_input_embeddings().weight.norm(dim=-1).mean().detach().clone())

    def read(self, sender, doc_ids, doc_mask):
        if self.gate is not None:
            self.gate["on"] = True
        if self.read_mode == "slot":
            b = doc_ids.size(0)
            x = torch.cat([sender.get_input_embeddings()(doc_ids), self.slots.unsqueeze(0).expand(b, -1, -1)], dim=1)
            mask = torch.cat([doc_mask, doc_mask.new_ones(b, self.k)], dim=1)
            out = sender.base_model(inputs_embeds=x, attention_mask=mask, output_hidden_states=True)
            if self.gate is not None:
                self.gate["on"] = False
            return torch.cat([F.normalize(out.hidden_states[i][:, -self.k:], dim=-1) * out.hidden_states[i].size(-1) ** 0.5
                              for i in self.layers], dim=-1)
        with torch.set_grad_enabled(self.gate is not None and torch.is_grad_enabled()):
            out = sender.base_model(input_ids=doc_ids, attention_mask=doc_mask, output_hidden_states=True)
        if self.gate is not None:
            self.gate["on"] = False
        # 층마다 크기가 다르므로 각각 고정 정규화한 뒤 이어붙인다
        return torch.cat([F.normalize(out.hidden_states[i], dim=-1) * out.hidden_states[i].size(-1) ** 0.5
                          for i in self.layers], dim=-1)

    def local_bias(self, pos, doc_mask):
        length = doc_mask.sum(1, keepdim=True).clamp(min=1).float()
        u = (pos.float() + 0.5) / length
        c = (torch.arange(self.k, device=pos.device).float() + 0.5) / self.k
        bias = -0.5 * ((u.unsqueeze(1) - c.view(1, -1, 1)) * self.k) ** 2
        bias = bias.masked_fill(doc_mask.unsqueeze(1) == 0, -1e4)
        return bias.repeat_interleave(self.heads, dim=0)

    @torch.no_grad()
    def resize(self, new_k):
        """질의 개수를 바꾼다. 기존 질의를 순서 방향으로 선형 보간해 새 개수로 만든다(구간 편향은 k로 다시 계산됨).

        벡터 32개는 매번 금방 오르지만 16개 이하는 학습 출발이 불안정해서(PLAN A·D), 32개로 출발해 줄여 가는 용량 커리큘럼에 쓴다.
        """
        q = self.queries.data.T.unsqueeze(0)  # (1, z, k)
        q = F.interpolate(q, size=new_k, mode="linear", align_corners=True)[0].T.contiguous()
        self.queries = nn.Parameter(q)
        self.k = new_k

    def to_z(self, h, doc_mask):
        if self.read_mode == "slot":
            x = self.in_proj(h)
            return self.slot_norm(x + self.slot_ff(x))
        pos = (doc_mask.cumsum(1) - 1).clamp(min=0)
        x = self.in_proj(h) + sinusoid(pos, self.z_dim)
        bias = self.local_bias(pos, doc_mask)
        q = self.queries.unsqueeze(0).expand(h.size(0), -1, -1)
        for blk in self.blocks:
            q = blk(q, x, bias)
        return q

    def to_receiver(self, z):
        for blk in self.dec_blocks:
            z = blk(z)
        return F.normalize(self.decoder(z), dim=-1) * self.scale

    def forward(self, sender, doc_ids, doc_mask):
        return self.to_receiver(self.to_z(self.read(sender, doc_ids, doc_mask), doc_mask))

    align_target = PoolChannel.align_target


@torch.no_grad()
def evaluate(channel, s_tok, sender, r_tok, receiver, docs, device, batch, controls=True):
    """full / swapped(다른 문서의 메시지) / zero(0으로 채움) / none(메시지 없음) 정답률과 복원 토큰 정확도."""
    channel.eval()
    msgs = torch.cat([channel(sender, *encode_docs(s_tok, [d.text for d in docs[s: s + batch]], device))
                      for s in range(0, len(docs), batch)])
    items = [(di, qi) for di, d in enumerate(docs) for qi in range(len(d.pairs))]
    names = ("full", "swapped", "zero", "none") if controls else ("full",)
    acc, recon = {n: [] for n in names}, []
    for s in range(0, len(items), batch):
        chunk = items[s: s + batch]
        di = torch.tensor([c[0] for c in chunk])
        prompts = [qa_prompt(docs[d], q) for d, q in chunk]
        targets = [" " + docs[d].answer(q) for d, q in chunk]
        variants = {"full": msgs[di], "swapped": msgs[(di + 1) % len(docs)], "zero": torch.zeros_like(msgs[di]), "none": None}
        for n in names:
            acc[n].append(exact_match(*receiver_targets(r_tok, receiver, variants[n], prompts, targets, device)).cpu())
    for s in range(0, len(docs), batch):
        chunk = docs[s: s + batch]
        recon.append(token_hits(*receiver_targets(r_tok, receiver, msgs[s: s + batch], [RECON_PROMPT] * len(chunk),
                                                  [" " + d.text for d in chunk], device)).cpu())
    channel.train()
    out = {n: torch.cat(v).mean().item() for n, v in acc.items()}
    out["recon_tok"] = torch.cat(recon).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="결과 파일 이름에 붙는 꼬리표")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="송신자 모델")
    ap.add_argument("--receiver", default=None, help="수신자 모델. 기본은 송신자와 같은 모델")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-schedule", default=None,
                    help="용량 커리큘럼. 예: 32:1500,16:4500 = 32개로 1500스텝 뒤 16개로 4500스텝. 주면 --k·--steps 무시")
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--layers", default=None, help="읽을 송신자 층, 쉼표 구분. 기본은 가운데 층 하나")
    ap.add_argument("--pool-layers", type=int, default=1, help="풀러 블록 수. 2 이상이면 질의 간 어텐션이 켜진다")
    ap.add_argument("--dec-attn", type=int, default=0, help="디코더 앞 슬롯 간 어텐션 블록 수")
    ap.add_argument("--sender-lora", type=int, default=0, help="송신자 LoRA 랭크. 0이면 송신자 동결")
    ap.add_argument("--zero-gate", action="store_true", help="추가 블록을 0 게이트로 감싸 기준선과 같은 상태에서 출발")
    ap.add_argument("--read", choices=["pool", "slot"], default="pool", help="pool: 토큰 상태 어텐션 풀링 / slot: 슬롯 임베딩")
    ap.add_argument("--warmup-steps", type=int, default=800)
    ap.add_argument("--warmup-batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--task-lr", type=float, default=1e-4)
    ap.add_argument("--task-lr-ramp", type=int, default=0,
                    help="과제 학습 첫 N스텝 동안 학습률을 task-lr/10에서 task-lr로 선형 증가. 0이면 끔")
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-docs", type=int, default=64, help="중간 평가에 쓰는 문서 수(학습·보류 각각). 마지막 평가는 보류 전체")
    ap.add_argument("--seed", type=int, default=0, help="초기화·배치 순서 시드")
    ap.add_argument("--data-seed", type=int, default=0, help="문서 생성 시드. 시드 반복 시 데이터는 고정")
    args = ap.parse_args()

    phases = [(int(a), int(b)) for a, b in (x.split(":") for x in args.k_schedule.split(","))] if args.k_schedule \
        else [(args.k, args.steps)]
    args.k, args.steps = phases[0][0], sum(n for _, n in phases)
    torch.manual_seed(args.seed)
    device = pick_device()
    tok, model = load_frozen(args.model, device)
    sender = model
    if args.receiver in (None, args.model):
        r_tok = AutoTokenizer.from_pretrained(args.model)  # encode_docs가 패딩 방향을 바꾸므로 수신자용은 따로 둔다
        if r_tok.pad_token_id is None:
            r_tok.pad_token = r_tok.eos_token
        receiver = model
    else:
        # 2단계: 서로 다른 모델. 송신자 은닉 상태를 수신자 입력 임베딩 공간으로 번역해야 한다
        r_tok, receiver = load_frozen(args.receiver, device)

    gate, lora_params = None, []
    if args.sender_lora > 0:
        gate = {"on": False}
        lora_params = add_sender_lora(model, args.sender_lora, gate)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [model.config.num_hidden_layers // 2]
    channel = DeepChannel(sender, receiver, args.k, args.z_dim, layers, args.pool_layers, args.dec_attn, gate=gate,
                          zero_gate=args.zero_gate, read_mode=args.read).to(device)
    params = list(channel.parameters()) + lora_params
    n_train_params = sum(p.numel() for p in params)

    train = make_docs(args.n_train, seed=args.data_seed)
    heldout = make_docs(args.n_heldout, seed=args.data_seed + 10_000)
    seen = {d.text for d in train}
    heldout = [d for d in heldout if d.text not in seen]
    print(f"sender={args.model} receiver={args.receiver or args.model}", flush=True)
    print(f"tag={args.tag} device={device} k={args.k} z={args.z_dim} layers={layers} pool_layers={args.pool_layers} "
          f"read={args.read} dec_attn={args.dec_attn} zero_gate={args.zero_gate} sender_lora={args.sender_lora} trainable={n_train_params/1e6:.2f}M "
          f"train={len(train)} heldout={len(heldout)}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)
    history, t0 = [], time.time()

    def sample(n):
        return [train[i] for i in torch.randint(len(train), (n,), generator=gen).tolist()]

    def aligned(batch_docs):
        msg = channel(sender, *encode_docs(tok, [d.text for d in batch_docs], device))
        r_ids, r_mask = encode_docs(r_tok, [d.text for d in batch_docs], device)
        return msg, align_loss(msg, channel.align_target(receiver, r_ids, r_mask))

    def report(step, extra):
        tr = evaluate(channel, tok, sender, r_tok, receiver, train[: args.eval_docs], device, args.batch, controls=False)
        ho = evaluate(channel, tok, sender, r_tok, receiver, heldout[: args.eval_docs], device, args.batch)
        row = {"step": step, "sec": time.time() - t0, "train_full": tr["full"], "train_recon_tok": tr["recon_tok"],
               **{f"heldout_{k}": v for k, v in ho.items()}, **extra}
        history.append(row)
        print(f"step {step:5d}  k={channel.k:<3d} train {tr['full']:.3f}  heldout {ho['full']:.3f} / swapped {ho['swapped']:.3f} / "
              f"zero {ho['zero']:.3f} / none {ho['none']:.3f}  recon_tok {ho['recon_tok']:.3f}  {row['sec']:.0f}s", flush=True)

    for step in range(1, args.warmup_steps + 1):
        _, al = aligned(sample(args.warmup_batch))
        opt.zero_grad(set_to_none=True)
        al.backward()
        opt.step()
        if step % 100 == 0 or step == args.warmup_steps:
            print(f"warmup {step:4d}  align {al.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    for g in opt.param_groups:
        g["lr"] = args.task_lr
    report(0, {"k": channel.k})

    bounds, acc_steps = {}, 0
    for i, (k_i, n_i) in enumerate(phases):
        bounds[acc_steps + 1] = (i, k_i)
        acc_steps += n_i
    phase_start = 1
    for step in range(1, args.steps + 1):
        if step in bounds and bounds[step][0] > 0:
            k_new = bounds[step][1]
            channel.resize(k_new)
            params = list(channel.parameters()) + lora_params
            opt = torch.optim.AdamW(params, lr=args.task_lr, weight_decay=0.0)
            phase_start = step
            print(f"--- step {step}: 벡터 수 {k_new}로 변경", flush=True)
        if args.task_lr_ramp:
            frac = min(1.0, (step - phase_start + 1) / args.task_lr_ramp)
            for g in opt.param_groups:
                g["lr"] = args.task_lr * (0.1 + 0.9 * frac)
        batch_docs = sample(args.batch)
        qi = [int(torch.randint(len(d.pairs), (1,), generator=gen)) for d in batch_docs]
        msg, al = aligned(batch_docs)
        qa = ce_loss(*receiver_targets(r_tok, receiver, msg, [qa_prompt(d, q) for d, q in zip(batch_docs, qi)],
                                      [" " + d.answer(q) for d, q in zip(batch_docs, qi)], device))
        rec = ce_loss(*receiver_targets(r_tok, receiver, msg, [RECON_PROMPT] * len(batch_docs),
                                       [" " + d.text for d in batch_docs], device))
        loss = qa + args.recon_weight * rec + args.align_weight * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            gates = [round(m.g.item(), 4) for m in channel.modules() if isinstance(m, Gated)]
            report(step, {"k": channel.k, "qa_loss": qa.item(), "recon_loss": rec.item(), **({"gates": gates} if gates else {})})
            if gates:
                print(f"            gates {gates}", flush=True)

    final = evaluate(channel, tok, sender, r_tok, receiver, heldout, device, args.batch)
    print("최종(보류 전체 %d문서)  full %.3f / swapped %.3f / zero %.3f / none %.3f  recon_tok %.3f"
          % (len(heldout), final["full"], final["swapped"], final["zero"], final["none"], final["recon_tok"]), flush=True)
    out = ROOT / "results" / f"stage1_{args.tag}"
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "layers": layers, "trainable_params": n_train_params,
                                                    "history": history, "final_heldout": final}, indent=2, ensure_ascii=False))
    torch.save({"channel": channel.state_dict(), "lora": [p.detach().cpu() for p in lora_params], "args": vars(args)},
               out.with_suffix(".pt"))
    print(f"저장: {out}.json, {out}.pt")


if __name__ == "__main__":
    main()
