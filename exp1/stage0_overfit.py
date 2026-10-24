"""PLAN 0단계 — 과적합 시험.

문서 몇 개만으로 슬롯 임베딩·인코더·디코더를 학습해 학습 정답률이 100% 근처까지 가는지 본다.
가설 검증이 아니라 배관 검증: 여기서 안 오르면 코드가 틀린 것.

    python3 exp1/stage0_overfit.py                 # 기본: Qwen2.5-0.5B 동종, 문서 32개, k=16, Z=512
    python3 exp1/stage0_overfit.py --steps 20      # 연기 시험
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.tasks.keyvalue import make_docs, value_bits  # noqa: E402

RECON_PROMPT = "\nRepeat the document:"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen(name, device):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class Channel(nn.Module):
    """송신자 은닉 상태 -> 인코더 -> Z -> 디코더 -> 수신자 입력 임베딩."""

    def __init__(self, sender, receiver, k, z_dim, hidden=1024):
        super().__init__()
        d_s, d_r = sender.config.hidden_size, receiver.config.hidden_size
        s_emb = sender.get_input_embeddings().weight
        r_emb = receiver.get_input_embeddings().weight
        self.k = k
        self.slots = nn.Parameter(torch.randn(k, d_s) * s_emb.std().item())
        self.encoder = mlp(d_s, hidden, z_dim)
        self.decoder = mlp(z_dim, hidden, d_r)
        # 수신자 토큰 임베딩의 평균 크기로 시작. 안 맞추면 동결된 수신자가 벡터를 잡음으로 취급
        self.scale = nn.Parameter(r_emb.norm(dim=-1).mean().detach().clone())
        # 슬롯 은닉 상태는 크기가 수백이고 문서 간 코사인이 0.96이라(진단 A), 공통 성분을 빼고 크기를 맞춘 뒤 인코더에 넣는다
        self.h_center = nn.Parameter(torch.zeros(k, d_s))
        self.register_buffer("h_std", torch.ones(()))

    @torch.no_grad()
    def calibrate(self, sender, doc_ids, doc_mask):
        """문서 묶음으로 슬롯별 평균과 전체 표준편차를 잡는다. 학습 시작 전에 한 번."""
        h = self.read(sender, doc_ids, doc_mask)
        self.h_center.copy_(h.mean(0))
        self.h_std.copy_((h - h.mean(0, keepdim=True)).std())

    def read(self, sender, doc_ids, doc_mask):
        """문서 뒤에 슬롯 k개를 붙여 통과시키고, 슬롯 위치의 마지막 층 은닉 상태를 꺼낸다. doc_ids는 왼쪽 패딩."""
        b = doc_ids.size(0)
        emb = sender.get_input_embeddings()(doc_ids)
        x = torch.cat([emb, self.slots.unsqueeze(0).expand(b, -1, -1)], dim=1)
        mask = torch.cat([doc_mask, doc_mask.new_ones(b, self.k)], dim=1)
        h = sender.base_model(inputs_embeds=x, attention_mask=mask).last_hidden_state
        return h[:, -self.k:, :]

    def to_z(self, h):
        return self.encoder((h - self.h_center) / self.h_std)

    def to_receiver(self, z):
        return F.normalize(self.decoder(z), dim=-1) * self.scale

    def forward(self, sender, doc_ids, doc_mask):
        return self.to_receiver(self.to_z(self.read(sender, doc_ids, doc_mask)))


def sinusoid(pos, dim):
    """pos: (b, t) 정수 위치 -> (b, t, dim) 사인 위치 부호."""
    freq = torch.exp(-torch.arange(0, dim, 2, device=pos.device) * (9.21 / dim))
    ang = pos.unsqueeze(-1).float() * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PoolChannel(nn.Module):
    """송신자(기울기 없음)의 문서 토큰 은닉 상태 전체를 학습 질의 k개가 어텐션으로 모아 Z를 만든다.

    슬롯 방식은 동결된 송신자가 슬롯 위치에 문서 내용을 모으도록 슬롯 임베딩만으로 유도해야 해서
    0단계에서 문서별 정보가 실리지 않았다(results/stage0.json). 풀링은 토큰 위치의 상태를 직접 본다.
    """

    def __init__(self, sender, receiver, k, z_dim, layer, hidden=1024, heads=8):
        super().__init__()
        d_s, d_r = sender.config.hidden_size, receiver.config.hidden_size
        self.k, self.layer, self.z_dim = k, layer, z_dim
        self.in_proj = nn.Linear(d_s, z_dim)
        self.queries = nn.Parameter(torch.randn(k, z_dim) * 0.02)
        self.attn = nn.MultiheadAttention(z_dim, heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(z_dim), nn.LayerNorm(z_dim)
        self.ff = mlp(z_dim, hidden, z_dim)
        self.decoder = mlp(z_dim, hidden, d_r)
        self.scale = nn.Parameter(receiver.get_input_embeddings().weight.norm(dim=-1).mean().detach().clone())

    @torch.no_grad()
    def read(self, sender, doc_ids, doc_mask):
        out = sender.base_model(input_ids=doc_ids, attention_mask=doc_mask, output_hidden_states=True)
        return out.hidden_states[self.layer]

    def to_z(self, h, doc_mask):
        pos = (doc_mask.cumsum(1) - 1).clamp(min=0)  # 왼쪽 패딩이므로 실제 토큰 기준 위치
        # 첫 토큰 위치는 크기가 1000을 넘는다. 학습 가능한 LayerNorm은 MPS에서 기울기가 inf가 돼서 고정 정규화를 쓴다
        h = F.normalize(h, dim=-1) * h.size(-1) ** 0.5
        x = self.in_proj(h) + sinusoid(pos, self.z_dim)
        q = self.queries.unsqueeze(0).expand(h.size(0), -1, -1)
        a, _ = self.attn(q, x, x, attn_mask=self.local_bias(pos, doc_mask), need_weights=False)
        z = self.norm1(q + a)
        return self.norm2(z + self.ff(z))

    def local_bias(self, pos, doc_mask):
        """질의 j가 문서의 j번째 구간을 주로 보도록 하는 어텐션 편향 + 패딩 차단. (b*heads, k, t)

        편향 없이 시작하면 모든 질의가 문서 전체를 고르게 평균해 문서 간 차이가 사라진다.
        """
        length = doc_mask.sum(1, keepdim=True).clamp(min=1).float()
        u = (pos.float() + 0.5) / length                                   # (b, t) 토큰의 상대 위치 0~1
        c = (torch.arange(self.k, device=pos.device).float() + 0.5) / self.k  # (k,) 질의의 구간 중심
        bias = -0.5 * ((u.unsqueeze(1) - c.view(1, -1, 1)) * self.k) ** 2
        bias = bias.masked_fill(doc_mask.unsqueeze(1) == 0, -1e4)
        return bias.repeat_interleave(self.attn.num_heads, dim=0)

    def to_receiver(self, z):
        return F.normalize(self.decoder(z), dim=-1) * self.scale

    def forward(self, sender, doc_ids, doc_mask):
        return self.to_receiver(self.to_z(self.read(sender, doc_ids, doc_mask), doc_mask))

    @torch.no_grad()
    def align_target(self, receiver, r_ids, r_mask):
        """정렬 예열의 목표: 슬롯 j에 배정된 문서 토큰들의 수신자 임베딩 평균. (b, k, d_r)

        토큰은 상대 위치가 가장 가까운 슬롯에 배정한다. 토큰이 하나도 없는 슬롯(k > 토큰 수)은 가장 가까운 토큰을 쓴다.
        동결된 수신자는 자기 토큰 임베딩 근처의 벡터만 읽으므로, 무작위로 시작한 채널을 먼저 그 근처로 옮겨 놓는다.
        """
        pos = (r_mask.cumsum(1) - 1).clamp(min=0)
        u = (pos.float() + 0.5) / r_mask.sum(1, keepdim=True).clamp(min=1).float()
        c = (torch.arange(self.k, device=r_ids.device).float() + 0.5) / self.k
        dist = (u.unsqueeze(1) - c.view(1, -1, 1)).abs()                  # (b, k, t)
        dist = dist.masked_fill(r_mask.unsqueeze(1) == 0, 1e4)
        assign = F.one_hot(dist.argmin(1), self.k).transpose(1, 2).float()  # 토큰 -> 가장 가까운 슬롯
        assign = assign * r_mask.unsqueeze(1)
        nearest = F.one_hot(dist.argmin(2), r_ids.size(1)).float()          # 슬롯 -> 가장 가까운 토큰
        empty = assign.sum(2, keepdim=True) == 0
        w = torch.where(empty, nearest, assign)
        w = w / w.sum(2, keepdim=True)
        return w @ receiver.get_input_embeddings()(r_ids)


def align_loss(msg, target):
    return (1 - F.cosine_similarity(msg, target, dim=-1)).mean()


def encode_docs(tok, texts, device):
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_receiver_batch(tok, receiver, msg, prompts, targets, device):
    """[메시지 벡터] + [프롬프트] + [타깃]. 손실은 타깃 위치에서만."""
    emb_layer = receiver.get_input_embeddings()
    p_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    t_ids = [tok(t, add_special_tokens=False).input_ids for t in targets]
    k = 0 if msg is None else msg.size(1)
    width = k + max(len(p) + len(t) for p, t in zip(p_ids, t_ids))
    b = len(prompts)
    ids = torch.full((b, width - k), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((b, width), -100, dtype=torch.long)
    mask = torch.zeros((b, width), dtype=torch.long)
    for i, (p, t) in enumerate(zip(p_ids, t_ids)):
        n = len(p) + len(t)
        ids[i, :n] = torch.tensor(p + t)
        labels[i, k + len(p): k + n] = torch.tensor(t)
        mask[i, : k + n] = 1
    x = emb_layer(ids.to(device))
    if msg is not None:
        x = torch.cat([msg, x], dim=1)
    return x, mask.to(device), labels.to(device)


def receiver_targets(tok, receiver, msg, prompts, targets, device):
    """타깃 위치에서만 로짓을 계산한다. 전 위치 x 전 어휘 로짓은 배치 16에서 수 GB라 맥에서 스왑이 난다."""
    x, mask, labels = build_receiver_batch(tok, receiver, msg, prompts, targets, device)
    h = receiver.base_model(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, :-1, :]
    labels = labels[:, 1:]
    sel = labels != -100
    logits = receiver.get_output_embeddings()(h[sel])
    rows = sel.nonzero()[:, 0]
    return logits, labels[sel], rows, labels.size(0)


def ce_loss(logits, gold, rows, b):
    return F.cross_entropy(logits, gold)


def token_hits(logits, gold, rows, b):
    return (logits.argmax(-1) == gold).float()


def exact_match(logits, gold, rows, b):
    """타깃 토큰 전부가 argmax와 일치하는 예시의 비율. 전부 일치하면 그리디 생성 결과도 같다."""
    miss = torch.zeros(b, device=logits.device).index_add_(0, rows, 1.0 - token_hits(logits, gold, rows, b))
    return (miss == 0).float()


def qa_prompt(doc, i):
    # 답 앞머리를 문서 문장과 같은 꼴로 준다. "Answer:"로 끝내면 수신자가 문서 원문을 직접 봐도
    # 정답률이 0.15밖에 안 나오고(상한이 낮음), 이 꼴이면 0.99다.
    return f"\nQuestion: {doc.question(i)}\nAnswer: Code {doc.pairs[i][0]} has value"


@torch.no_grad()
def evaluate(channel, s_tok, sender, r_tok, receiver, docs, device, batch=16):
    """문서마다 모든 질문을 물어본다. full / shuffled(다른 문서의 메시지) / none(메시지 없음)."""
    channel.eval()
    msgs = []
    for s in range(0, len(docs), batch):
        ids, mask = encode_docs(s_tok, [d.text for d in docs[s: s + batch]], device)
        msgs.append(channel(sender, ids, mask))
    msgs = torch.cat(msgs)
    items = [(di, qi) for di, d in enumerate(docs) for qi in range(len(d.pairs))]
    acc = {"full": [], "shuffled": [], "none": []}
    recon = []
    for s in range(0, len(items), batch):
        chunk = items[s: s + batch]
        di = torch.tensor([c[0] for c in chunk])
        prompts = [qa_prompt(docs[d], q) for d, q in chunk]
        targets = [" " + docs[d].answer(q) for d, q in chunk]
        for name, m in (("full", msgs[di]), ("shuffled", msgs[(di + 1) % len(docs)]), ("none", None)):
            acc[name].append(exact_match(*receiver_targets(r_tok, receiver, m, prompts, targets, device)).cpu())
    for s in range(0, len(docs), batch):
        chunk = docs[s: s + batch]
        recon.append(token_hits(*receiver_targets(r_tok, receiver, msgs[s: s + batch], [RECON_PROMPT] * len(chunk),
                                                  [" " + d.text for d in chunk], device)).cpu())
    channel.train()
    out = {name: torch.cat(v).mean().item() for name, v in acc.items()}
    out["recon_token_acc"] = torch.cat(recon).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--receiver", default=None, help="기본은 송신자와 같은 모델")
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--read", choices=["pool", "slot"], default="pool", help="pool: 토큰 상태 어텐션 풀링 / slot: 슬롯 임베딩")
    ap.add_argument("--layer", type=int, default=None, help="pool에서 읽을 송신자 층. 기본은 가운데 층")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--warmup-steps", type=int, default=500, help="정렬 예열 스텝 수 (pool 전용). 0이면 생략")
    ap.add_argument("--align-weight", type=float, default=1.0, help="예열 뒤 과제 손실에 더할 정렬 손실 가중치")
    ap.add_argument("--task-lr", type=float, default=1e-4,
                    help="예열 뒤 학습률. 예열과 같은 1e-3으로 두면 첫 수십 스텝에 정렬이 깨져 정답률이 0으로 돌아간다")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "stage0.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    s_tok, sender = load_frozen(args.sender, device)
    if args.receiver in (None, args.sender):
        r_tok, receiver = AutoTokenizer.from_pretrained(args.sender), sender
        if r_tok.pad_token_id is None:
            r_tok.pad_token = r_tok.eos_token
    else:
        r_tok, receiver = load_frozen(args.receiver, device)

    docs = make_docs(args.n_docs, n_pairs=args.n_pairs, seed=args.seed)
    if args.read == "pool":
        if args.layer is None:
            args.layer = sender.config.num_hidden_layers // 2
        channel = PoolChannel(sender, receiver, args.k, args.z_dim, args.layer).to(device)
    else:
        channel = Channel(sender, receiver, args.k, args.z_dim).to(device)
        channel.calibrate(sender, *encode_docs(s_tok, [d.text for d in docs], device))
    n_train = sum(p.numel() for p in channel.parameters())
    print(f"device={device} sender={args.sender} receiver={args.receiver or args.sender}")
    print(f"docs={len(docs)} pairs/doc={args.n_pairs} value_bits/doc={value_bits(args.n_pairs):.1f} "
          f"read={args.read} layer={args.layer} k={args.k} z={args.z_dim} trainable={n_train/1e6:.2f}M")
    print(f"예시 문서: {docs[0].text}")

    opt = torch.optim.AdamW(channel.parameters(), lr=args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)
    history, t0 = [], time.time()

    def aligned(batch_docs):
        """메시지와 정렬 손실. 수신자 순전파가 없어 빠르다."""
        ids, mask = encode_docs(s_tok, [d.text for d in batch_docs], device)
        msg = channel(sender, ids, mask)
        r_ids, r_mask = encode_docs(r_tok, [d.text for d in batch_docs], device)
        return msg, align_loss(msg, channel.align_target(receiver, r_ids, r_mask))

    use_align = args.read == "pool"
    if use_align and args.warmup_steps > 0:
        for step in range(1, args.warmup_steps + 1):
            _, al = aligned(docs)
            opt.zero_grad(set_to_none=True)
            al.backward()
            opt.step()
            if step % 50 == 0 or step == args.warmup_steps:
                print(f"warmup {step:4d}  align {al.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        ev = evaluate(channel, s_tok, sender, r_tok, receiver, docs, device, args.batch)
        history.append({"step": 0, "align_loss": al.item(), "sec": time.time() - t0, **ev})
        for g in opt.param_groups:
            g["lr"] = args.task_lr
        print(f"예열 직후   acc full {ev['full']:.3f} / shuffled {ev['shuffled']:.3f} / none {ev['none']:.3f}  "
              f"recon_tok {ev['recon_token_acc']:.3f}", flush=True)
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(docs), (args.batch,), generator=gen).tolist()
        qi = [int(torch.randint(args.n_pairs, (1,), generator=gen)) for _ in idx]
        batch_docs = [docs[i] for i in idx]
        if use_align:
            msg, al = aligned(batch_docs)
        else:
            ids, mask = encode_docs(s_tok, [d.text for d in batch_docs], device)
            msg, al = channel(sender, ids, mask), 0.0

        qa = ce_loss(*receiver_targets(r_tok, receiver, msg, [qa_prompt(d, q) for d, q in zip(batch_docs, qi)],
                                      [" " + d.answer(q) for d, q in zip(batch_docs, qi)], device))
        rec = ce_loss(*receiver_targets(r_tok, receiver, msg, [RECON_PROMPT] * len(batch_docs),
                                       [" " + d.text for d in batch_docs], device))
        loss = qa + args.recon_weight * rec + args.align_weight * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(channel.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(channel, s_tok, sender, r_tok, receiver, docs, device, args.batch)
            row = {"step": step, "qa_loss": qa.item(), "recon_loss": rec.item(), "sec": time.time() - t0, **ev}
            history.append(row)
            print(f"step {step:5d}  qa {qa.item():.3f}  recon {rec.item():.3f}  "
                  f"acc full {ev['full']:.3f} / shuffled {ev['shuffled']:.3f} / none {ev['none']:.3f}  "
                  f"recon_tok {ev['recon_token_acc']:.3f}  {row['sec']:.0f}s", flush=True)
            if ev["full"] >= 0.99:
                break

    final = history[-1] if history else {}
    passed = final.get("full", 0) >= 0.95 and final.get("shuffled", 1) <= 0.10
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "trainable_params": n_train, "history": history,
                                          "passed": passed}, indent=2, ensure_ascii=False))
    print(f"\n0단계 관문(full >= 0.95, shuffled <= 0.10): {'통과' if passed else '미통과'}  -> {args.out}")


if __name__ == "__main__":
    main()
