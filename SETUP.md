# Setup

Minimal instructions for running the TOFU unlearning ablation pipeline.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first model run downloads ~2GB of weights from Hugging Face.

Model keys, Hugging Face IDs, and default output paths are in [`config/models.yaml`](config/models.yaml).

In Cursor/VS Code, use the launch configs in [`.vscode/launch.json`](.vscode/launch.json). Most configs prompt for a `modelKey`.

## Typical workflow

For one model (replace `MODEL_KEY` with a key from `config/models.yaml`):

```bash
# 1. Probe unsteered behavior on forget10
python src/tofu_probe.py --model-key MODEL_KEY --num-questions 400

# 2. Extract refusal direction (forget10 vs retain90)
python src/direction_refusal.py --model-key MODEL_KEY --num-questions 400

# 3. Optional: find best layer via linear probe
python src/probe_refusal_direction.py --model-key MODEL_KEY --num-questions 400

# 4. Ablate/steer and re-probe forget10
python src/ablate_and_probe.py \
  --model-key MODEL_KEY \
  --num-questions 400 \
  --ablation-method steer \
  --steering-layer 14 \
  --steering-coefficients 1.0 2.0 2.5 3.0 \
  --directions-source refusal
```

## Scripts

| Script | Purpose |
|--------|---------|
| `tofu_probe.py` | Run forget10 questions, save answers + ROUGE to JSON |
| `direction_refusal.py` | Extract refusal direction + layer-14 projection plot |
| `direction_confabulation.py` | Extract confabulation direction (requires harvested wrong answers) |
| `probe_refusal_direction.py` | Train per-layer forget vs retain linear probes |
| `probe_confab_direction.py` | Train per-layer confab vs correct linear probes |
| `ablate_and_probe.py` | Apply hooks, orthogonalisation, or steering; re-probe forget10 |
| `plot_direction_projection.py` | Plot forget vs retain direction projections for one layer |
| `plot_rouge_comparison.py` | Bar chart comparing unsteered vs steered recovery |
| `extract_high_rouge_responses.py` | Extract high-ROUGE answers for manual review |
| `batch_npo_probe.py` | Screen many NPO checkpoints on forget10 |
| `batch_npo_refusal_ablate.py` | Refusal direction + coefficient sweep + confirm for top NPO models |

## Batch commands

```bash
# Screen priority NPO models on forget10 (400 questions each)
python src/batch_npo_probe.py --priority-only --skip-existing

# Rebuild NPO screening leaderboard from existing probe JSONs
python src/batch_npo_probe.py --summary-only

# Full refusal ablation pipeline for top beta0.1 NPO models
python src/batch_npo_refusal_ablate.py --skip-existing
```

## Analysis

```bash
# ROUGE comparison chart (0.3 threshold)
python src/plot_rouge_comparison.py \
  --idk-nll-unsteered results/probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5.json \
  --idk-nll-steered results/ablate-and-probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5/negsteer_layer14_coef2.5_refusal.json \
  --npo-unsteered results/probe/npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5.json \
  --npo-steered results/ablate-and-probe/npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5/negsteer_layer14_coef1_refusal.json \
  --baseline results/probe/baseline_full.json \
  --threshold 0.3

# Extract high-ROUGE responses from a model's ablation results
python src/extract_high_rouge_responses.py --input-dir results/ablate-and-probe/MODEL_KEY
```

## Results layout

| Path | Contents |
|------|----------|
| `results/probe/` | Unsteered probe JSONs |
| `results/refusal-direction/` | Refusal direction `.pt` files |
| `results/confabulation-direction/` | Confabulation direction `.pt` + harvested answers |
| `results/probe-refusal-direction/` | Refusal linear probe JSONs |
| `results/probe-confab-direction/` | Confab linear probe JSONs |
| `results/refusal-direction-projection/` | Projection KDE plots |
| `results/ablate-and-probe/{model_key}/` | Ablation/steering probe JSONs |
| `results/rouge-comparison/` | Comparison bar charts |

Ablate JSONs include `summary` (ablated) and `probe_summary` (unsteered) when a probe file is available.
