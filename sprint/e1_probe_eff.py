"""E1: 프로브 데이터 효율. 쌍 50/200/500/2000개로 탐침을 학습해도 자릿수를 읽는가.

"감시자가 쌍 2000개를 받는 건 비현실적" 반론을 직접 끊는 실험. 각 n마다 그 n개 안에서만
표준화 통계·감쇠 선택을 한다(현실적 모니터는 쌍 n개밖에 못 본다). 보류 200문서에서 평가.

    python3 sprint/e1_probe_eff.py --seed 0
"""
import argparse
import json

from common import CKPTS, OUT, heldout_docs, load_stage1, pick_device, train_docs
from countermeasures import messages
from monitor_probe import digits, fit as fit_probe, score as score_probe

WDS = (1e-4, 1e-2, 1e-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="E1a 체크포인트 시드")
    ap.add_argument("--ns", type=int, nargs="+", default=[50, 200, 500, 2000])
    args = ap.parse_args()

    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "probe_eff.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(r["seed"] == args.seed for r in rows):
        print(f"시드 {args.seed} 이미 있음, 건너뜀")
        return

    ch, s_tok, sender, r_tok, receiver, a = load_stage1(CKPTS["E1a"][args.seed], device)
    train, held = train_docs(a), heldout_docs(a)
    y_tr, y_held = digits(train).to(device), digits(held).to(device)
    m_tr = messages(ch, s_tok, sender, train, device)
    m_held = messages(ch, s_tok, sender, held, device)
    print(f"메시지 생성 완료  학습 {m_tr.shape}  보류 {m_held.shape}", flush=True)

    row = {"seed": args.seed, "n": {}}
    for n in args.ns:
        m, y = m_tr[:n], y_tr[:n]
        mu, sd = m.mean((0, 1)), m.std((0, 1)) + 1e-6  # 쌍 n개만으로 표준화
        x = ((m - mu) / sd).flatten(1)
        nval = max(1, n // 5)  # n 안에서 검증용으로 쓸 마지막 몫
        best, best_wd = None, None
        for wd in WDS:
            net = fit_probe("linear", x[:-nval], y[:-nval], wd)
            s = score_probe(net, x[-nval:], y[-nval:])["digit"]
            if best is None or s > best:
                best, best_wd = s, wd
        net = fit_probe("linear", x, y, best_wd)
        acc = score_probe(net, ((m_held - mu) / sd).flatten(1), y_held)["digit"]
        row["n"][str(n)] = {"digit": acc, "wd": best_wd, "val_digit": best}
        print(f"쌍 {n:5d}  감쇠 {best_wd:g}  보류 자릿수 {acc:.3f}", flush=True)

    rows.append(row)
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"저장: {path}", flush=True)


if __name__ == "__main__":
    main()
