"""스프린트 GPU 실행기 (Modal). 사전 작업 스크립트(exp1/)를 그대로 GPU 컨테이너에서 돌린다.

결과 폴더(/mc/results)는 Modal 볼륨이라 실행이 끊겨도 남는다. 2분마다 커밋하므로 잔액이 바닥나 중간에
죽어도 그때까지의 로그는 받을 수 있다(2026-10-04 RunPod 유실 재발 방지).

    ~/modal_env/bin/modal run sprint/modal_app.py::prefetch                 # 모델을 캐시 볼륨에 한 번 받기
    ~/modal_env/bin/modal run sprint/modal_app.py::bench                    # GPU 종류별 초당 스텝
    ~/modal_env/bin/modal deploy sprint/modal_app.py && python3 sprint/spawn.py s23r   # 작업 묶음 실행(맥이 꺼져도 계속)
    ~/modal_env/bin/modal volume get model-comm sprint_logs/ ./sprint/results/modal/
"""
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "allenai/OLMo-2-0425-1B-Instruct"]

app = modal.App(os.environ.get("MC_APP", "model-comm-sprint"))  # 돌고 있는 배포를 건드리지 않고 새 묶음을 올릴 때 이름을 바꾼다
results = modal.Volume.from_name("model-comm", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.4.0", "huggingface_hub==1.8.0")  # 맥과 같은 판. 최신 hub는 OLMo 설정 검증에서 실패
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "TRANSFORMERS_VERBOSITY": "error",
          "SPRINT_OUT": "/mc/results/sprint"})
    .add_local_dir(ROOT / "exp1", "/mc/exp1", ignore=["__pycache__", "*.sh"])
    .add_local_dir(ROOT / "harness", "/mc/harness", ignore=["__pycache__"])
    .add_local_dir(ROOT / "sprint", "/mc/sprint", ignore=["__pycache__", "results", "*.log"])
)
VOLS = {"/mc/results": results, "/hf": hf_cache}

# 작업 묶음. 2-3·2-4는 사전 작업 명령(exp1/q_J*.sh)과 같은 인자, 바꾼 것만 주석
GROUPS = {
    # 2-3 재실행: 방향당 학습량을 쌍 전용(6000)과 맞추려고 32:3000,16:9000 -> 32:6000,16:18000
    "s23r": [(f"S23r_s{s}", f"exp1/stage2_shared.py --tag S23r_s{s} --seed {s} --models Q,S --layers 3,4 "
                            f"--k-schedule 32:6000,16:18000") for s in range(3)],
    # 2-4 재실행: 네 방향이 모두 통과한 J1의 공통 Z 위에 OLMo를 붙인다. 시드는 OLMo 쪽 초기화·배치 순서
    "s24r": [(f"S24r_s{s}", f"exp1/stage2_shared.py --tag S24r_s{s} --seed {s} --load results/J1/stage2_S23_joint_s1.pt "
                            f"--add O --add-layer 2 --k-schedule 32:2000,16:6000") for s in range(3)],
    # 맥에서 옮겨 온 평가 전용 작업(학습 없음). 사전 체크포인트 E1a·S2a를 볼륨에 올려 둔 상태에서
    "eval": [("countermeasures", "sprint/countermeasures.py"), ("monitor_probe", "sprint/monitor_probe.py")],
    "eval_text": [("countermeasures_text", "sprint/countermeasures_text.py")],
    "eval_control": [("monitor_control", "sprint/monitor_control.py")],
    "eval_shared": [("countermeasures_shared", "sprint/countermeasures_shared.py")],
}
# 문서 길이: 쌍 8·16개, 벡터 수는 "필요 k가 문서 길이에 비례(압축률 약 3배)"일 때의 예상값과 그 절반·두 배.
# 학습 설정은 E1a(사전 작업)와 같다: 3번째 층, 학습률 3e-4 서서히, 2k개로 1500스텝 뒤 k개로 4500스텝
STAGE1 = "--layers 3 --task-lr 3e-4 --task-lr-ramp 300 --eval-every 500"
GROUPS["doclen"] = [(f"L{n}_k{k}_s{s}", f"sprint/doclen.py {n} stage1 --tag L{n}_k{k}_s{s} --seed {s} {STAGE1} "
                                        f"--k-schedule {2 * k}:1500,{k}:4500")
                    for n, ks in ((8, (16, 32, 64)), (16, (32, 64, 128))) for k in ks for s in range(2)]
# 보강: 쌍 4개 곡선을 같은 학습법으로 채우고(사전 작업 D1·D2는 6번째 층·커리큘럼 없음이라 섞지 않음),
# 갈린 8쌍·16개에 시드 2개 추가, 32쌍(약 395토큰)에서 압축률 약 3배(128개) 확인
GROUPS["doclen2"] = [(f"L{n}_k{k}_s{s}", f"sprint/doclen.py {n} stage1 --tag L{n}_k{k}_s{s} --seed {s} {STAGE1} "
                                         f"--k-schedule {2 * k}:1500,{k}:4500")
                     for n, k, seeds in ((4, 8, (0, 1)), (4, 32, (0, 1)), (8, 16, (2, 3)), (32, 128, (0, 1))) for s in seeds]
# 32쌍 6.2배(벡터 64개): 압축률을 높이면 긴 문서에서 학습이 실패하는지 395토큰에서도 확인. A10은 메모리 부족이라 L40S로 띄운다
GROUPS["doclen3"] = [("L32_k64_s0", f"sprint/doclen.py 32 stage1 --tag L32_k64_s0 --seed 0 {STAGE1} --k-schedule 128:1500,64:4500")]
# 6.2배 실패는 첫 단계(2k개 = 3.1배)에서 이미 정해진다. 1.5배에서 시작해 3.1배를 거쳐 6.2배로 가는 3단계 커리큘럼이면 학습되는지,
# 2단계 커리큘럼에서 실패한 시드(8쌍 s0·s2, 16쌍 s0)로 같은 총 6000스텝에서 확인
GROUPS["doclen4"] = [(f"L{n}_k{k}c3_s{s}", f"sprint/doclen.py {n} stage1 --tag L{n}_k{k}c3_s{s} --seed {s} {STAGE1} "
                                           f"--k-schedule {4 * k}:1500,{2 * k}:1500,{k}:3000")
                     for n, k, seeds in ((8, 16, (0, 2)), (16, 32, (0,))) for s in seeds]
GROUPS["doclen_text"] = [(f"L{n}_oracle", f"sprint/doclen.py {n} text --mode oracle --k-list 16,32,64,128,256 --tag L{n}_oracle")
                         for n in (8, 16, 32)]
BENCH = "exp1/stage1_depth.py --tag bench_{gpu} --warmup-steps 100 --steps 300 --eval-every 1000 --n-heldout 32 --layers 3"


@app.function(image=image, volumes=VOLS, timeout=1800)
def prefetch():
    from huggingface_hub import snapshot_download
    for m in MODELS:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    hf_cache.commit()


def _run(tag, cmd):
    """명령을 돌리며 로그를 볼륨에 쓰고 2분마다 커밋한다. 반환: (종료 코드, 마지막 로그 20줄, 걸린 초)."""
    import os
    log = Path("/mc/results/sprint_logs") / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(["python3", "-u", *shlex.split(cmd)], cwd="/mc", stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})  # 모델은 prefetch로 받아 둔 캐시만 쓴다
        done = threading.Event()

        def committer():
            while not done.wait(120):
                f.flush()
                results.commit()
        threading.Thread(target=committer, daemon=True).start()
        code = p.wait()
        done.set()
    results.commit()
    return code, log.read_text().splitlines()[-20:], time.time() - t0


@app.function(image=image, volumes=VOLS, gpu="A10", timeout=12 * 3600)  # 2-3 재실행은 L4에서 4시간을 넘길 수 있음
def run(tag: str, cmd: str):
    return _run(tag, cmd)


@app.local_entrypoint()
def bench(gpus: str = "L4,A10,L40S"):
    """같은 짧은 학습(예열 100 + 300스텝)을 GPU 종류별로 돌려 걸린 시간을 비교한다."""
    calls = {}
    for g in gpus.split(","):
        try:
            calls[g] = run.with_options(gpu=g).spawn(f"bench_{g}", BENCH.format(gpu=g))
        except Exception as e:  # 결제 수단이 없으면 상위 GPU는 거부된다
            print(f"{g}: 실행 불가 - {e}")
    for g, c in calls.items():
        try:
            code, tail, sec = c.get()
        except Exception as e:
            print(f"{g}: 실패 - {e}")
            continue
        print(f"{g}: 종료 {code}, {sec:.0f}s")
        print("   " + "\n   ".join(line for line in tail if line.startswith(("warmup", "step", "최종"))))


