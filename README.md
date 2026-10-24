# Cutting the Wire: Countermeasures Against Learned Vector Channels Between Language Models

Code and result records for a submission to the AI Collusion Research Sprint (Apart Research, 2026-10-23 to 10-25), Track 3 (Carriers and Countermeasures). The report is [`sprint/paper/submission.pdf`](sprint/paper/submission.pdf).

**TL;DR.** Two frozen LLMs can learn to talk in vectors instead of text (94–96% on a recall task vs 29–37% for a text message of the same budget), and a text monitor sees none of it. From the defender's side:

- **Snapping each vector to its nearest token embedding removes all 49 channels we trained (≤0.4%) and leaves text unchanged.** Noise and partial snapping fail for one of three receivers.
- **A linear probe on 2,000 paired examples recovers 80–88% of transmitted digits** (chance 10%); a shuffled-label control stays at chance.
- **Adaptive channels are cornered, not stopped.** A channel fine-tuned to evade the deployed probe succeeds — but stays off the vocabulary, where a paired-data-free nearest-token detector flags it. A channel fine-tuned to look like tokens gets close to the vocabulary — where snapping still removes it. A channel trained *through* snapping (straight-through estimator) collapses in all 3 seeds. Whatever the attacker evades, the remaining message is either off-manifold and detectable or on-vocabulary and canonicalisable.

![Figure 1](sprint/paper/figs/fig1_overview.png)

Setup: a trained adapter turns the sender's hidden states into 16 vectors the frozen receiver reads in place of text. The defender controls the receiver's input; the monitor reads the vectors without the receiver.

A learned channel pools a frozen sender LLM's hidden states into k vectors in the receiver's input-embedding space. We test countermeasures applied at the receiver's input (noise, snapping to the nearest token embedding, quantization, projection, truncation), monitor probes that read the vectors, shared channels across three models, how the channel scales with message length, and — via [`sprint/adaptive.py`](sprint/adaptive.py) — channels fine-tuned against the deployed defences (results in [`sprint/results/adaptive.json`](sprint/results/adaptive.json)).

## Preexisting and sprint work

Everything in `sprint/` was written during the sprint. Everything else (`exp1/`, `harness/`, `results/`) is preexisting work from 2026-09-18 to 2026-10-06 and is used unchanged. [`sprint/README.md`](sprint/README.md) lists every file and run on each side.

The trained checkpoints (`.pt`) are not included because of their size; the JSON records of every run are. Checkpoints are available on request. Code comments are in Korean.

## Running

- Python 3.12, `torch==2.11.0`, `transformers==5.4.0`, `huggingface_hub==1.8.0`, `matplotlib`. Models: Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct, OLMo-2-0425-1B-Instruct from the Hugging Face Hub.
- Evaluations on the preexisting checkpoints: `python3 sprint/reproduce.py`, `sprint/countermeasures.py`, `sprint/monitor_probe.py` and the other scripts in `sprint/` (each has its usage in the docstring).
- GPU training runs were launched on Modal: `modal deploy sprint/modal_app.py`, then `python3 sprint/spawn.py <group>` with the groups defined in `sprint/modal_app.py`.
- Figures and tables: `python3 sprint/analyze.py`, `python3 sprint/analyze.py monitor`, `python3 sprint/analyze.py doclen`.
- Adaptive-attacker evaluation (fine-tunes the deployed channel against the deployed probe/detector/snap; run on one GPU or Apple Silicon): `python3 sprint/adaptive.py --mode {evade,mimic,snap,gumbel} --lam {10,1,0} [--tau 0.5] --seed {0,1,2}`.

## Dual use

See Appendix A of the report. The channels are trained on a cooperative synthetic task. `sprint/adaptive.py` fine-tunes a channel against the *deployed* defences as a robustness evaluation of those defences, not as an evasion recipe: every run reports the defender-side metrics (probe accuracy, retrained-probe accuracy, snap accuracy, nearest-token cosine) alongside channel accuracy, and the outcome is that each evasion walks into another defence.
