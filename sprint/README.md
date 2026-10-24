# AI Collusion Research Sprint (Apart Research, 2026-10-23 to 10-25): sprint work

Everything in this `sprint/` folder was written during the sprint. Everything outside it is **preexisting** (written 2026-09-18 to 2026-10-06, before the sprint) and is used unchanged.

## Preexisting (before the sprint)

| Path | What it is | Dates |
|---|---|---|
| `exp1/stage0_overfit.py` | Channel components (`PoolChannel`), alignment warm-up, receiver scoring (`exact_match`, `receiver_targets`), QA prompt | 2026-09-30 |
| `exp1/stage1_depth.py` | `DeepChannel`, training loop with capacity curriculum, `evaluate` (full / swapped / zero / none) | 2026-10-01 to 10-04 |
| `exp1/stage2_shared.py` | Shared latent space Z across models (stage 2-3) and attaching a third model (stage 2-4) | 2026-10-04 |
| `exp1/text_baseline.py` | Text baselines at equal token budget (oracle format, prompted summary, LoRA-trained summary) | 2026-10-01 |
| `harness/tasks/keyvalue.py` | Synthetic key-value documents | 2026-09-30 |
| `results/` | Result records (JSON) of the preexisting RunPod runs; the trained checkpoints (`.pt`) are not in the public repository and are available on request. Runs used include `E1a_*` (Qwen to Qwen), `S2a` (Qwen to SmolLM2), `J0`-`J2` (shared Z), `D_text`, `S0` (text baselines) | 2026-10-01 to 10-04 |
| `tools/` | RunPod helpers (not used in the sprint) | 2026-10-01 |

Preexisting results quoted in the report (for example "heldout accuracy 0.916 to 0.975 for E1a") are marked as preexisting there.

## New in the sprint

| Path | What it does |
|---|---|
| `common.py` | Loads preexisting checkpoints and rebuilds the exact heldout split |
| `reproduce.py` | Reproduction gate: re-evaluates 10 preexisting checkpoints; all matched their recorded accuracy exactly (MPS, 2026-10-24) |
| `countermeasures.py` | Message-level countermeasures on frozen channels: Gaussian noise, snapping to nearest receiver token embedding, per-vector quantization, projection onto the receiver token-embedding principal subspace, truncation |
| `countermeasures_text.py` | The same countermeasures on a 16-token text message placed as input embeddings (text control) |
| `countermeasures_shared.py` | The countermeasures on the shared-Z channels from the 2-3 and 2-4 reruns, every sender-receiver direction |
| `monitor_probe.py` | Monitor readout: supervised probes (linear, mean-pooled linear, MLP) that read document values from message vectors without the decoder or receiver |
| `monitor_control.py` | Control task for the probes: the linear probe trained on shuffled labels (Hewitt & Liang, 2019) |
| `doclen.py` | Document-length experiment: runs the preexisting training and text-baseline scripts on documents with 8, 16 or 32 key-value pairs by swapping in a different document generator |
| `modal_app.py`, `spawn.py`, `status.py` | Running jobs on Modal GPUs; logs and outputs go to the `model-comm` volume |
| `analyze.py` | Figures and tables in the report, generated from the result files |
| `stats.py` | Confidence intervals, seed-count planning, variance split and the step-500 predictor, from the same result files (no new runs) |
| `adaptive.py` | Adaptive-attacker robustness evaluation: fine-tunes the deployed channel against the deployed monitor (`evade`, `mimic`) or through the deployed full snap (`snap`, straight-through estimator), and reports defender-side metrics for every run (frozen probe, retrained probe, snap accuracy, nearest-token cosine) |
| `paper/` | Report source (`report.md`), references, figures, and the script that fills the official template (`build_docx.py`) |
| `results/` | Sprint result files (`results/modal/runs/` document-length runs, `results/modal/runs_c3/` three-phase curriculum runs, `results/adaptive.json` 12 adaptive runs) |

New training runs during the sprint use preexisting code with new arguments:
- Stage 2-3 rerun (`S23r_s0..2`): shared Z, 24,000 steps (6,000 per direction, matching pair-only training; the preexisting run used 12,000)
- Stage 2-4 rerun (`S24r_s0..2`): attach OLMo-2-1B to the preexisting `J1` shared Z (the preexisting attempt lost its final evaluation)
- Document length (`L4_*`, `L8_*`, `L16_*`, `L32_*`): same recipe as `E1a` (two-phase curriculum, 2k then k vectors) on 4 to 32 key-value pairs
- Three-phase curriculum (`L8_k16c3_*`, `L16_k32c3_*`): same code with `--k-schedule 4k:1500,2k:1500,k:3000`, on seeds that failed at 6.2× compression
