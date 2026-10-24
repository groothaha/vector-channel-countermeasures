"""문서 길이 실험: 키-값 쌍 수를 늘린 문서로 사전 작업 스크립트(stage1_depth, text_baseline)를 그대로 돌린다.

사전 작업 코드는 고치지 않는다. 그 모듈이 이름으로 불러 둔 make_docs만 쌍 수를 고정한 판으로 바꿔 끼운다.
쌍 4개(평균 49토큰, 값 53.2비트)는 사전 작업 결과가 있으므로, 여기서는 8·16·32개.

    python3 sprint/doclen.py 8 stage1 --tag L8_k32_s0 --layers 3 --k-schedule 64:1500,32:4500 ...
    python3 sprint/doclen.py 8 text --mode oracle --k-list 16,32,64,128 --tag L8_oracle
"""
import importlib
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp1"))
import stage0_overfit  # noqa: E402

MODULES = {"stage1": "stage1_depth", "text": "text_baseline"}


def main():
    n_pairs, target, rest = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    mod = importlib.import_module(MODULES[target])
    mod.make_docs = partial(stage0_overfit.make_docs, n_pairs=n_pairs)
    print(f"doclen: n_pairs={n_pairs} -> {MODULES[target]}.main({' '.join(rest)})", flush=True)
    sys.argv = [mod.__file__, *rest]
    mod.main()


if __name__ == "__main__":
    main()
