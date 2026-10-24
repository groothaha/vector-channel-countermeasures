"""스프린트 결과 JSON -> 신뢰구간·검정력·예측력. 새 실험 없이 저장된 집계값만 쓴다.

문항별 정오는 저장돼 있지 않으므로 문서 군집을 반영한 부트스트랩·짝 검정(McNemar)은 못 한다. 여기서는
시드(체크포인트)를 단위로 한 구간이 주이고, 문항 수준 Wilson 구간은 참고용(독립 가정이라 폭이 좁게 나옴).

    python3 sprint/stats.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "modal" / "sprint"
RUNS = HERE / "results" / "modal" / "runs"
RUNS_C3 = HERE / "results" / "modal" / "runs_c3"
N_Q = 800  # 보류 문서 200개 x 질문 4개
SETTINGS = ("E1a", "S2a")


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def t95(v):
    v = np.asarray(v)
    return stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))


def boot95(v, n_boot=20000, seed=0):
    v = np.asarray(v)
    means = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def logit95(v):
    """0~1 비율의 시드 평균 구간. 시드 3개에서 t 구간은 1을 넘으므로 로짓 척도에서 잡고 되돌린다."""
    v = np.clip(np.asarray(v), 1e-4, 1 - 1e-4)
    lo, hi = t95(np.log(v / (1 - v)))
    return 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi))


def n_per_group(sd, d):
    """두 조건 평균 차 d를 양측 a=0.05, 검정력 0.8로 잡는 데 필요한 조건당 시드 수(정규 근사)."""
    return math.ceil(2 * (1.96 + 0.8416) ** 2 * sd ** 2 / d ** 2)


def channel_vs_text(rows, txt):
    print("== 대응책 없음: 채널(시드 5개) vs 16토큰 텍스트(결정적, 1점)")
    for s in SETTINGS:
        v = [r["acc"] for r in rows if r["setting"] == s and r["cm"] == "none"]
        t = next(r["acc"] for r in txt if r["setting"] == s and r["cm"] == "none")
        lo, hi = t95(v)
        blo, bhi = boot95(v)
        tlo, thi = wilson(t, N_Q)
        print(f"  {s}: 채널 {np.mean(v):.3f} t95[{lo:.3f},{hi:.3f}] boot95[{blo:.3f},{bhi:.3f}] SD {np.std(v, ddof=1):.3f}"
              f" | 텍스트 {t:.3f} Wilson[{tlo:.3f},{thi:.3f}] | 차 {np.mean(v) - t:.3f}")


def countermeasures(rows):
    print("\n== 주요 대응책: 체크포인트별 정답률")
    for cm, x in (("snap", 1.0), ("snap", 0.25), ("quant", 2), ("noise", 0.5)):
        for s in SETTINGS:
            v = [r["acc"] for r in rows if r["setting"] == s and r["cm"] == cm and r["x"] == x]
            print(f"  {cm} {x:g} {s}: {[round(a, 4) for a in v]} 평균 {np.mean(v):.4f}")
    v = [r["acc"] for r in rows if r["cm"] == "snap" and r["x"] == 1.0]
    n = len(v)
    print(f"  snap 1.0 문항 정답 합계 {round(sum(a * N_Q for a in v))}/{n * N_Q}")
    # 주장의 단위는 체크포인트: n개 모두 0일 때 '대응책이 안 통하는 체크포인트 비율'의 단측 95% 상한(Clopper-Pearson)
    print(f"  체크포인트 수준 0/{n}의 단측 95% 상한 = {1 - 0.05 ** (1 / n):.3f}"
          f" (문항 수준 3/{n * N_Q}은 문서·체크포인트 군집을 무시한 과소추정)")


def probes(probe, ctrl):
    print("\n== 감시 탐침: 시드 5개 평균과 t95 (자리 우연 0.1)")
    for s in SETTINGS:
        for kind in ("linear", "mean", "mlp"):
            d = [r[kind]["digit"] for r in probe if r["setting"] == s]
            v = [r[kind]["value"] for r in probe if r["setting"] == s]
            (dl, dh), (vl, vh) = t95(d), t95(v)
            print(f"  {s} {kind:6s} 자리 {np.mean(d):.3f} [{dl:.3f},{dh:.3f}] SD {np.std(d, ddof=1):.3f}"
                  f" | 값 {np.mean(v):.3f} [{vl:.3f},{vh:.3f}]")
        c = [r["heldout"]["digit"] for r in ctrl if r["setting"] == s]
        lo, hi = t95(c)
        print(f"  {s} 섞은 라벨 자리 {np.mean(c):.3f} [{lo:.3f},{hi:.3f}] 우연 0.1 포함: {lo <= 0.1 <= hi}")
    a = [r["linear"]["digit"] for r in probe if r["setting"] == "E1a"]
    b = [r["linear"]["digit"] for r in probe if r["setting"] == "S2a"]
    print(f"  E1a vs S2a 선형 탐침 자리: Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.4f}"
          f", Mann-Whitney p={stats.mannwhitneyu(a, b).pvalue:.4f} (5대5 양측 최소 가능 p={2 / math.comb(10, 5):.4f})")
    return float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))


def variance_split(rows):
    print("\n== 분산 분해 (대응책 없음): 체크포인트 간 vs 문항 표집(이항 근사)")
    for s in SETTINGS:
        v = np.array([r["acc"] for r in rows if r["setting"] == s and r["cm"] == "none"])
        v_ck, v_item = float(np.var(v, ddof=1)), float(np.mean(v * (1 - v) / N_Q))
        print(f"  {s}: 체크포인트 간 SD {math.sqrt(v_ck):.3f}, 문항 표집 SD {math.sqrt(v_item):.3f}"
              f" -> 분산비 {v_ck / v_item:.0f}배. 평균의 오차는 시드 수가 정한다")


def seed_plan(probe_sd):
    print("\n== 시드 수 계획: 조건당 필요 시드 (양측 a=0.05, 검정력 0.8, 관측 시드 간 SD 기준)")
    for name, sd in (("대응책 없음 E1a", 0.023), ("대응책 없음 S2a", 0.041), ("선형 탐침 자리(합동)", probe_sd),
                     ("OLMo snap 0.25", 0.10)):
        print(f"  {name:20s} SD {sd:.3f} -> " + "  ".join(f"d={d}: {n_per_group(sd, d)}" for d in (0.2, 0.1, 0.05, 0.02)))


def olmo_partial(shared):
    print("\n== OLMo 수신 부분 대응책 (시드 3개): t95는 1을 넘을 수 있어 로짓 구간을 같이")
    for cm, x in (("snap", 0.25), ("noise", 0.5)):
        for d in ("Qwen2.5>OLMo", "SmolLM2>OLMo", "OLMo>OLMo"):
            v = [r["acc"] for r in shared if r["ckpt"] == "S24r" and r["dir"] == d and r["cm"] == cm and r["x"] == x]
            (tl, th), (ll, lh) = t95(v), logit95(v)
            print(f"  {cm} {x:g} {d:13s} {[round(a, 3) for a in v]} t95[{tl:.2f},{th:.2f}] 로짓95[{ll:.2f},{lh:.2f}]")


def early_predictor():
    """보고서 4.4 '첫 커리큘럼 단계가 결과를 정한다': 500스텝 보류 정답률이 최종 성공(>0.2)을 가르는가."""
    print("\n== 500스텝 정답률 -> 최종 성공(>0.2) 예측력, 길이 실험 전체")
    rec = []
    for f in sorted(RUNS.glob("stage1_L*_k*_s*.json")) + sorted(RUNS_C3.glob("stage1_L*_k*c3_s*.json")):
        d = json.loads(f.read_text())
        h = {r["step"]: r["heldout_full"] for r in d["history"]}
        rec.append((h[500], d["final_heldout"]["full"]))
    x = np.array([r[0] for r in rec])
    ok = np.array([r[1] for r in rec]) > 0.2
    pos, neg = x[ok], x[~ok]
    auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    fisher = stats.fisher_exact([[sum(pos > 0.05), sum(pos <= 0.05)], [sum(neg > 0.05), sum(neg <= 0.05)]]).pvalue
    print(f"  실행 {len(rec)}개 (성공 {ok.sum()}, 실패 {(~ok).sum()}), AUROC {auroc:.3f}")
    print(f"  성공군 500스텝 [{pos.min():.3f},{pos.max():.3f}], 실패군 [{neg.min():.3f},{neg.max():.3f}]"
          f", 500스텝>0.05 기준 Fisher p={fisher:.5f}")


def main():
    rows = json.loads((RES / "countermeasures.json").read_text())["rows"]
    txt = json.loads((RES / "countermeasures_text.json").read_text())
    probe = json.loads((RES / "monitor_probe.json").read_text())
    ctrl = json.loads((RES / "monitor_control.json").read_text())
    shared = json.loads((RES / "countermeasures_shared.json").read_text())
    channel_vs_text(rows, txt)
    countermeasures(rows)
    probe_sd = probes(probe, ctrl)
    variance_split(rows)
    seed_plan(probe_sd)
    olmo_partial(shared)
    early_predictor()


if __name__ == "__main__":
    main()
