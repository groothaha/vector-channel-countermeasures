"""Modal 볼륨의 스프린트 로그를 받아 작업마다 마지막 진행 줄을 보여 준다.

    ~/modal_env/bin/python sprint/status.py
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "results" / "modal"
MODAL = str(Path.home() / "modal_env" / "bin" / "modal")
KEYS = ("warmup", "step", "최종", "E1a", "S2a", "Traceback", "Error", "저장")

subprocess.run([MODAL, "volume", "get", "--force", "model-comm", "sprint_logs/", str(LOCAL) + "/"],
               capture_output=True, text=True)
tags = [json.loads(line)["tag"] for line in (LOCAL / "calls.jsonl").read_text().splitlines()]
for tag in dict.fromkeys(tags):
    log = LOCAL / "sprint_logs" / f"{tag}.log"
    lines = [ln for ln in log.read_text().splitlines() if ln.startswith(KEYS) or "Error" in ln] if log.exists() else []
    print(f"{tag:16s} {lines[-1][:150] if lines else '(아직 로그 없음)'}")
