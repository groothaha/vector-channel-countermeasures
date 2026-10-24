"""배포된 Modal 앱에 작업 묶음을 띄우고 바로 끝난다. 작업은 맥과 무관하게 돈다. 로그는 볼륨 sprint_logs/.

    ~/modal_env/bin/modal deploy sprint/modal_app.py
    ~/modal_env/bin/python sprint/spawn.py eval s23r s24r
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_app import GROUPS  # noqa: E402

run = modal.Function.from_name(os.environ.get("MC_APP", "model-comm-sprint"), "run")
book = Path(__file__).resolve().parent / "results" / "modal" / "calls.jsonl"
book.parent.mkdir(parents=True, exist_ok=True)
fn = run.with_options(gpu=os.environ["MC_GPU"]) if os.environ.get("MC_GPU") else run  # 예: 메모리가 큰 L40S
only = set(os.environ["MC_ONLY"].split(",")) if os.environ.get("MC_ONLY") else None  # 묶음 중 일부 태그만
for group in sys.argv[1:]:
    for tag, cmd in GROUPS[group]:
        if only and tag not in only:
            continue
        call = fn.spawn(tag, cmd)
        with book.open("a") as f:
            f.write(json.dumps({"tag": tag, "group": group, "call_id": call.object_id, "at": time.strftime("%F %T")}) + "\n")
        print(f"{tag}: {call.object_id}")
