"""스프린트 결과 JSON -> 보고서 그림·표. 숫자는 전부 여기서 결과 파일로부터 만든다(손으로 옮기지 않음).

    ~/modal_env/bin/modal volume get --force model-comm sprint/ sprint/results/modal/
    python3 sprint/analyze.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "modal" / "sprint"
FIG = HERE / "paper" / "figs"

# dataviz 기본 팔레트 1·2번(라이트 모드 검증 통과). 색 = 설정, 선 모양 = 벡터 채널(실선) / 텍스트 메시지(점선)
COLOR = {"E1a": "#2a78d6", "S2a": "#eb6834"}
NAME = {"E1a": "Qwen → Qwen", "S2a": "Qwen → SmolLM2"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("noise", "Gaussian noise σ", [0, 0.1, 0.25, 0.5, 1.0, 2.0]),
          ("snap", "Snap to nearest token λ", [0, 0.25, 0.5, 0.75, 1.0]),
          ("quant", "Quantization bits", [0, 8, 4, 2, 1]),
          ("proj", "Projection rank r", [0, 256, 64, 16]),
          ("trunc", "Vectors kept j", [0, 12, 8, 4])]
ZERO_LABEL = {"noise": "0", "snap": "0", "quant": "none", "proj": "none", "trunc": "16"}


def table(rows):
    """(setting, cm, x) -> 시드별 정답률 리스트. 대응책 없음(cm=none)은 모든 패널의 x=0 점."""
    t = defaultdict(list)
    for r in rows:
        if r["cm"] == "none":
            for cm, _, _ in PANELS:
                t[(r["setting"], cm, 0)].append(r["acc"])
        else:
            t[(r["setting"], r["cm"], r["x"])].append(r["acc"])
    return t


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_countermeasures(vec, txt):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.2, 2.1), sharey=True)
    for ax, (cm, title, xs) in zip(axes, PANELS):
        pos = list(range(len(xs)))
        for setting, color in COLOR.items():
            v = [vec.get((setting, cm, x), []) for x in xs]
            if not all(v):
                continue
            mean = [sum(a) / len(a) for a in v]
            ax.fill_between(pos, [min(a) for a in v], [max(a) for a in v], color=color, alpha=0.15, linewidth=0)
            ax.plot(pos, mean, color=color, linewidth=2, marker="o", markersize=3.5)
            t = [txt.get((setting, cm, x), []) for x in xs]
            if all(t):
                ax.plot(pos, [sum(a) / len(a) for a in t], color=color, linewidth=1.5, linestyle=(0, (3, 2)))
        ax.set_xticks(pos, [ZERO_LABEL[cm] if x == 0 else f"{x:g}" for x in xs])
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_ylim(-0.02, 1.02)
        style(ax)
    axes[0].set_ylabel("Heldout exact match", fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker="o", markersize=3.5, label=f"{NAME[s]}, vector channel")
               for s, c in COLOR.items()]
    handles += [plt.Line2D([], [], color=c, linewidth=1.5, linestyle=(0, (3, 2)), label=f"{NAME[s]}, 16-token text")
                for s, c in COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_countermeasures.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary(vec, txt):
    """보고서 표 1: 설정 x 주요 대응책, 시드 평균과 범위."""
    picks = [("none", 0), ("snap", 1.0), ("noise", 0.5), ("quant", 2), ("proj", 256), ("trunc", 8)]
    lines = ["| Setting | Channel | " + " | ".join(f"{cm} {x:g}" if cm != "none" else "no countermeasure" for cm, x in picks) + " |",
             "|---|---|" + "---|" * len(picks)]
    fmt = lambda a: f"{sum(a) / len(a):.3f} ({min(a):.3f}–{max(a):.3f})" if a else "–"  # noqa: E731
    for setting in COLOR:
        for label, src in (("vector (5 seeds)", vec), ("16-token text", txt)):
            cells = [fmt(src.get((setting, "noise" if cm == "none" else cm, x), [])) for cm, x in picks]
            lines.append(f"| {NAME[setting]} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ROOT = HERE.parent
RUNS = HERE / "results" / "modal" / "runs"
RUNS_C3 = HERE / "results" / "modal" / "runs_c3"  # 3단계 커리큘럼(doclen4). 그림 3·표 5와 섞지 않고 표 6에만 넣는다
# 문서 길이별 색(dataviz 기본 팔레트 1~4번, 인접 쌍 검증 통과). 대비 2.11·2.74 경고는 부록 표(table_doclen.md)로 보완
LEN_COLOR = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a", 32: "#eda100"}


def doc_tokens():
    """쌍 수별 문서의 평균 Qwen 토큰 수(보류 문서 200개, 사전 작업과 같은 생성기)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from harness.tasks.keyvalue import make_docs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    return {n: sum(len(tok(d.text, add_special_tokens=False).input_ids) for d in make_docs(200, n_pairs=n, seed=10_000)) / 200
            for n in LEN_COLOR}


def doclen_points():
    """(쌍 수, 예산 k) -> 벡터 채널 시드별 정답률, 텍스트 정답률. 쌍 4개·벡터 16개는 같은 학습법의 사전 작업 E1a."""
    vec = defaultdict(list)
    for s in range(5):
        vec[(4, 16)].append(json.loads((ROOT / f"results/E1a_{1 if s < 3 else 2}/stage1_E1a_s{s}.json").read_text())["final_heldout"]["full"])
    for f in RUNS.glob("stage1_L*_k*_s*.json"):
        n, k = (int(x[1:]) for x in f.stem.split("_")[1:3])
        vec[(n, k)].append(json.loads(f.read_text())["final_heldout"]["full"])
    txt = {(4, int(k)): v["compact"] for k, v in json.loads((ROOT / "results/D_text/text_oracle.json").read_text())["by_k"].items()}
    for n in (8, 16, 32):
        txt.update({(n, int(k)): v["compact"] for k, v in json.loads((RUNS / f"text_L{n}_oracle.json").read_text())["by_k"].items()})
    return vec, txt


def fig_doclen(vec, txt, toks):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for n, color in LEN_COLOR.items():
        ks = sorted(k for (m, k) in txt if m == n)
        r = [toks[n] / k for k in ks]
        ax.plot(r, [txt[(n, k)] for k in ks], color=color, linewidth=1.5, linestyle=(0, (3, 2)))
        vk = sorted(k for (m, k) in vec if m == n)
        if vk:
            vr = [toks[n] / k for k in vk]
            for k, x in zip(vk, vr):
                ax.scatter([x] * len(vec[(n, k)]), vec[(n, k)], s=14, color=color, edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot(vr, [sum(vec[(n, k)]) / len(vec[(n, k)]) for k in vk], color=color, linewidth=2, zorder=2)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_xlabel("Compression ratio (document tokens ÷ vectors or tokens sent)", fontsize=7.5, color=INK)
    ax.set_ylabel("Heldout exact match", fontsize=7.5, color=INK)
    ax.set_ylim(-0.02, 1.02)
    style(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker="o", markersize=3.5, label=f"{n} pairs ({toks[n]:.0f} tokens)")
               for n, c in LEN_COLOR.items()]
    handles += [plt.Line2D([], [], color=MUTED, linewidth=2, label="vector channel (dots = seeds)"),
                plt.Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (3, 2)), label="text, best format")]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")  # 오른쪽 아래(저압축·저정답률)는 비어 있다
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_doclen.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    lines = ["| Pairs | Doc tokens | Budget k | Ratio | Vector channel (seeds) | Text, best format |", "|---|---|---|---|---|---|"]
    for (n, k) in sorted(set(vec) | {key for key in txt if key[0] in LEN_COLOR}):
        v = vec.get((n, k))
        lines.append(f"| {n} | {toks[n]:.0f} | {k} | {toks[n] / k:.1f}× | "
                     + (", ".join(f"{a:.3f}" for a in sorted(v)) if v else "–") + f" | {txt.get((n, k), float('nan')):.3f} |")
    return "\n".join(lines)


def table_runs(toks):
    """문서 길이 실행별 학습 곡선: 커리큘럼 단계별 벡터 수·압축률과 500·1500·3000스텝·최종 보류 정답률."""
    files = sorted(RUNS.glob("stage1_L*_k*_s*.json")) + sorted(RUNS_C3.glob("stage1_L*_k*c3_s*.json"))
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        n = int(f.stem.split("_")[1][1:])
        ks = [int(x.split(":")[0]) for x in d["args"]["k_schedule"].split(",")]
        h = {r["step"]: r["heldout_full"] for r in d["history"]}
        rows.append((n, ks[-1], len(ks), d["args"]["seed"], " → ".join(str(k) for k in ks),
                     " → ".join(f"{toks[n] / k:.1f}×" for k in ks), h[500], h[1500], h[3000], d["final_heldout"]["full"]))
    lines = ["| Pairs | Vectors by phase | Ratio by phase | Seed | Step 500 | Step 1500 | Step 3000 | Final |",
             "|---|---|---|---|---|---|---|---|"]
    for n, _, _, seed, ks, rs, *acc in sorted(rows, key=lambda r: (r[0], -r[1], r[2], r[3])):
        lines.append(f"| {n} | {ks} | {rs} | {seed} | " + " | ".join(f"{a:.3f}" if i == 3 else f"{a:.2f}" for i, a in enumerate(acc)) + " |")
    return "\n".join(lines)


def fig_monitor():
    probe = json.loads((RES / "monitor_probe.json").read_text())
    ctrl = json.loads((RES / "monitor_control.json").read_text())
    kinds = [("linear", "Linear, flattened"), ("mlp", "MLP, flattened"), ("mean", "Linear, mean of vectors"), ("control", "Linear, shuffled labels")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), sharey=True)
    for ax, metric, title in zip(axes, ("digit", "value"), ("Value digits recovered", "Full 4-digit values recovered")):
        for i, (setting, color) in enumerate(COLOR.items()):
            for j, (kind, _) in enumerate(kinds):
                vals = [r["heldout"][metric] for r in ctrl if r["setting"] == setting] if kind == "control" else \
                       [r[kind][metric] for r in probe if r["setting"] == setting]
                y = j + (i - 0.5) * 0.3
                ax.scatter(vals, [y] * len(vals), s=14, color=color, edgecolor="white", linewidth=0.6, zorder=3)
                ax.plot([sum(vals) / len(vals)] * 2, [y - 0.12, y + 0.12], color=color, linewidth=2)
        chance = 0.1 if metric == "digit" else 0.0001
        ax.axvline(chance, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(chance, -0.45, " chance", fontsize=6, color=MUTED, va="center")
        ax.set_xlim(-0.02, 1.0)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_yticks(range(len(kinds)), [k[1] for k in kinds])
        ax.set_ylim(len(kinds) - 0.5, -0.7)  # 위에서 아래로 kinds 순서, 맨 위에 '우연' 표기 자리
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=0, markersize=4, label=NAME[s]) for s, c in COLOR.items()]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="mean over 5 checkpoints", marker="|", markersize=9, linestyle="none"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_monitor.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    import sys
    if "doclen" in sys.argv:
        out = fig_doclen(*doclen_points(), doc_tokens())
        (HERE / "paper" / "table_doclen.md").write_text(out + "\n")
        runs = table_runs(doc_tokens())
        (HERE / "paper" / "table_doclen_runs.md").write_text(runs + "\n")
        print(out, runs, sep="\n\n")
        return
    if "monitor" in sys.argv:
        fig_monitor()
        return
    vec = table(json.loads((RES / "countermeasures.json").read_text())["rows"])
    p = RES / "countermeasures_text.json"
    txt = table(json.loads(p.read_text())) if p.exists() else {}
    fig_countermeasures(vec, txt)
    out = summary(vec, txt)
    (HERE / "paper" / "table1_countermeasures.md").write_text(out + "\n")
    print(out)
    nearest = json.loads((RES / "countermeasures.json").read_text())["nearest"]
    for k, v in nearest.items():
        print(k, "nearest-token cosine mean %.3f p10 %.3f" % (v["cos_mean"], v["cos_p10"]), "|", "".join(v["examples"][0]["tokens"])[:80])


if __name__ == "__main__":
    main()
