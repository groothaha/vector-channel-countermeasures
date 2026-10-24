"""합성 키-값 문서. 모델이 사전 지식으로 답할 수 없으므로 맞힌 만큼이 채널로 넘어온 정보."""
import math
import random
from dataclasses import dataclass

KEY_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 헷갈리는 I/O/0/1 제외


@dataclass
class Doc:
    pairs: list  # [(key, value), ...]

    @property
    def text(self):
        return " ".join(f"Code {k} has value {v}." for k, v in self.pairs)

    def question(self, i):
        return f"What is the value of code {self.pairs[i][0]}?"

    def answer(self, i):
        return self.pairs[i][1]


def make_docs(n_docs, n_pairs=4, key_len=4, val_len=4, seed=0):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        keys = set()
        while len(keys) < n_pairs:
            keys.add("".join(rng.choice(KEY_CHARS) for _ in range(key_len)))
        keys = sorted(keys)  # set 순회 순서는 실행마다 달라지므로 정렬 후 섞는다
        rng.shuffle(keys)
        docs.append(Doc([(k, "".join(rng.choice("0123456789") for _ in range(val_len))) for k in keys]))
    return docs


def value_bits(n_pairs=4, val_len=4):
    """문서 하나의 값들이 담은 정보량(비트). 키는 질문에 주어지므로 제외."""
    return n_pairs * val_len * math.log2(10)
