"""재현 관문: 사전 작업 체크포인트를 다시 불러 평가해, 기록된 보류 정답률이 나오는지 본다.

여기서 기록값과 다르면(|차이| > 0.02) 이후 대응책·감시자 실험의 숫자는 의미가 없다.

    python3 sprint/reproduce.py
"""
import json
import time

from common import CKPTS, OUT, evaluate, heldout_docs, load_stage1, pick_device, recorded


def main():
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for setting, pts in CKPTS.items():
        for seed, pt in enumerate(pts):
            ch, s_tok, sender, r_tok, receiver, a = load_stage1(pt, device)
            docs = heldout_docs(a)
            ev = evaluate(ch, s_tok, sender, r_tok, receiver, docs, device, batch=16)
            rec = recorded(pt)
            row = {"setting": setting, "seed": seed, "k": ch.k, "n_docs": len(docs), "now": ev, "recorded": rec,
                   "diff_full": ev["full"] - rec["full"]}
            rows.append(row)
            print(f"{setting} s{seed} k={ch.k}  full {ev['full']:.3f} (기록 {rec['full']:.3f}, 차이 {row['diff_full']:+.3f})  "
                  f"swapped {ev['swapped']:.3f}  zero {ev['zero']:.3f}  none {ev['none']:.3f}  {time.time() - t0:.0f}s",
                  flush=True)
    passed = all(abs(r["diff_full"]) <= 0.02 for r in rows)
    (OUT / "reproduce.json").write_text(json.dumps({"device": device, "rows": rows, "passed": passed}, indent=2))
    print(f"재현 관문(|차이| <= 0.02 전부): {'통과' if passed else '미통과'}")


if __name__ == "__main__":
    main()
