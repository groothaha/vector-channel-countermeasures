"""스프린트(AI Collusion Research Sprint, 2026-10-23~25) 신규 코드의 공용부.

사전 작업(exp1/, harness/, results/ — 2026-09-30~10-06)의 채널 구조와 평가 함수를 고치지 않고 불러 쓴다.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SPRINT_OUT", ROOT / "sprint" / "results"))  # Modal에서는 볼륨 안
sys.path.insert(0, str(ROOT / "exp1"))
from stage0_overfit import encode_docs, load_frozen, make_docs, pick_device  # noqa: E402
from stage1_depth import DeepChannel, evaluate  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_models = {}


def model(name, device):
    """동결 모델을 한 번만 올린다. 같은 모델을 여러 체크포인트가 공유."""
    if name not in _models:
        _models[name] = load_frozen(name, device)
    return _models[name]


def receiver_tok(name):
    # stage1_depth와 같게 수신자용 토크나이저를 따로 둔다(encode_docs가 패딩 방향을 바꿈)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_stage1(pt, device):
    """stage1_depth.py가 저장한 체크포인트 -> (채널, 송신자 토크나이저, 송신자, 수신자 토크나이저, 수신자, 인자)."""
    ck = torch.load(pt, map_location="cpu")
    a = ck["args"]
    assert not ck["lora"], "송신자 LoRA 체크포인트는 다루지 않는다"
    s_tok, sender = model(a["model"], device)
    r_name = a.get("receiver") or a["model"]  # 초기 체크포인트(E1a)는 --receiver 인자가 생기기 전 것
    receiver = model(r_name, device)[1]
    r_tok = receiver_tok(r_name)
    layers = [int(x) for x in a["layers"].split(",")] if a["layers"] else [sender.config.num_hidden_layers // 2]
    k = ck["channel"]["queries"].shape[0]  # 용량 커리큘럼을 거쳤으면 마지막 벡터 수
    ch = DeepChannel(sender, receiver, k, a["z_dim"], layers, a.get("pool_layers", 1), a.get("dec_attn", 0),
                     zero_gate=a.get("zero_gate", False), read_mode=a.get("read", "pool"))
    ch.load_state_dict(ck["channel"])
    ch.to(device).eval().requires_grad_(False)
    return ch, s_tok, sender, r_tok, receiver, a


def heldout_docs(a):
    """stage1_depth.main()과 같은 보류 문서(학습 문서와 겹치는 것 제외)."""
    seen = {d.text for d in make_docs(a["n_train"], seed=a["data_seed"])}
    return [d for d in make_docs(a["n_heldout"], seed=a["data_seed"] + 10_000) if d.text not in seen]


def train_docs(a):
    return make_docs(a["n_train"], seed=a["data_seed"])


def recorded(pt):
    """같은 이름의 .json에 기록된 최종 보류 결과."""
    return json.loads(Path(pt).with_suffix(".json").read_text())["final_heldout"]


CKPTS = {
    "E1a": [ROOT / f"results/E1a_{1 if s < 3 else 2}/stage1_E1a_s{s}.pt" for s in range(5)],  # Qwen -> Qwen
    "S2a": [ROOT / f"results/S2a/stage1_S22_QS_s{s}.pt" for s in range(5)],                    # Qwen -> SmolLM2
}
