# vlaskool — Continual RL VLA

SmolVLA + LoRA + GRPO on ManiSkill, reproducing the methodology from:

> **"Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning"** (arXiv 2603.11653)

## Quick start

```bash
bash setup.sh
uv run python scripts/collect_demos.py --all-training
uv run python scripts/benchmark.py --dry-run
```

## Comparisons

| Mode | Description |
|---|---|
| `zero_shot` | Pretrained SmolVLA, no finetuning |
| `sft` | Sequential BC finetuning on seen tasks |
| `continual_rl` | Sequential GRPO + LoRA (paper's recipe) |

## Metrics

- **AVG**: final average success over training tasks (↑)
- **NBT**: negative backward transfer / forgetting (↓)
- **FWT**: forward transfer (↑)
- **ZS**: zero-shot success on held-out tasks (↑)
