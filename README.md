# Cutting the Wire: Countermeasures Against Learned Vector Channels Between Language Models

Code and result records for a submission to the AI Collusion Research Sprint (Apart Research, 2026-10-23 to 10-25), Track 3 (Carriers and Countermeasures). The report is [`sprint/paper/submission.pdf`](sprint/paper/submission.pdf).

A learned channel pools a frozen sender LLM's hidden states into k vectors in the receiver's input-embedding space. We test countermeasures applied at the receiver's input (noise, snapping to the nearest token embedding, quantization, projection, truncation), monitor probes that read the vectors, shared channels across three models, and how the channel scales with message length.

## Preexisting and sprint work

Everything in `sprint/` was written during the sprint. Everything else (`exp1/`, `harness/`, `results/`) is preexisting work from 2026-09-18 to 2026-10-06 and is used unchanged. [`sprint/README.md`](sprint/README.md) lists every file and run on each side.

The trained checkpoints (`.pt`) are not included because of their size; the JSON records of every run are. Checkpoints are available on request. Code comments are in Korean.

## Running

- Python 3.12, `torch==2.11.0`, `transformers==5.4.0`, `huggingface_hub==1.8.0`, `matplotlib`. Models: Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct, OLMo-2-0425-1B-Instruct from the Hugging Face Hub.
- Evaluations on the preexisting checkpoints: `python3 sprint/reproduce.py`, `sprint/countermeasures.py`, `sprint/monitor_probe.py` and the other scripts in `sprint/` (each has its usage in the docstring).
- GPU training runs were launched on Modal: `modal deploy sprint/modal_app.py`, then `python3 sprint/spawn.py <group>` with the groups defined in `sprint/modal_app.py`.
- Figures and tables: `python3 sprint/analyze.py`, `python3 sprint/analyze.py monitor`, `python3 sprint/analyze.py doclen`.

## Dual use

See Appendix A of the report. The channels here are non-adaptive and trained on a cooperative synthetic task; nothing in this repository trains a channel to evade a countermeasure or a monitor.
