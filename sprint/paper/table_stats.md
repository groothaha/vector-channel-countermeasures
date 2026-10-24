| Quantity | n | Estimate | 95% interval | Method |
|---|---|---|---|---|
| Vector channel, Qwen → Qwen, no countermeasure | 5 checkpoints | 0.955 | 0.926–0.983 | t over checkpoints |
| Vector channel, Qwen → SmolLM2, no countermeasure | 5 checkpoints | 0.940 | 0.889–0.991 | t over checkpoints |
| 16-token text, Qwen → Qwen | 800 questions | 0.373 | 0.340–0.407 | Wilson, questions independent |
| 16-token text, Qwen → SmolLM2 | 800 questions | 0.286 | 0.256–0.319 | Wilson, questions independent |
| Checkpoints with any direction above text after full snap | 16 checkpoints | 0 of 16 | upper bound 0.171 | one-sided Clopper–Pearson |
| Linear probe, digits recovered, Qwen → Qwen | 5 checkpoints | 0.875 | 0.866–0.884 | t over checkpoints |
| Linear probe, digits recovered, Qwen → SmolLM2 | 5 checkpoints | 0.803 | 0.727–0.878 | t over checkpoints |
| Shuffled-label control, Qwen → Qwen (chance 0.100) | 5 checkpoints | 0.099 | 0.091–0.108 | t over checkpoints |
| Shuffled-label control, Qwen → SmolLM2 (chance 0.100) | 5 checkpoints | 0.096 | 0.091–0.101 | t over checkpoints |
| Snap λ = 0.25, Qwen → OLMo | 3 seeds | 0.745 | 0.443–0.921 | t on logit scale |
| Snap λ = 0.25, SmolLM2 → OLMo | 3 seeds | 0.780 | 0.495–0.935 | t on logit scale |
| Snap λ = 0.25, OLMo → OLMo | 3 seeds | 0.828 | 0.767–0.877 | t on logit scale |
| Noise σ = 0.5, Qwen → OLMo | 3 seeds | 0.635 | 0.448–0.792 | t on logit scale |
| Noise σ = 0.5, SmolLM2 → OLMo | 3 seeds | 0.662 | 0.549–0.760 | t on logit scale |
| Noise σ = 0.5, OLMo → OLMo | 3 seeds | 0.741 | 0.577–0.861 | t on logit scale |
