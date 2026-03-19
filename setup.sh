#!/usr/bin/env bash
# setup.sh — bootstrap the vlaskool environment on an RTX 4090 system
# Run once: bash setup.sh

set -euo pipefail

echo "=== vlaskool setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'not detected')"
echo ""

# ── 1. Create uv virtualenv ─────────────────────────────────────────────────
echo "[1/5] Creating uv virtual environment (Python 3.11)..."
uv venv --python 3.11
echo "      venv created at .venv"

# ── 2. Install dependencies ─────────────────────────────────────────────────
echo "[2/5] Installing dependencies (this may take a few minutes)..."
uv pip install -e ".[dev]"

# Verify torch CUDA
echo "[3/5] Verifying PyTorch + CUDA..."
uv run python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  bfloat16:   {torch.cuda.is_bf16_supported()}')
"

# ── 4. Verify LeRobot / SmolVLA ─────────────────────────────────────────────
echo "[4/5] Checking LeRobot / SmolVLA availability..."
uv run python -c "
try:
    import lerobot
    print(f'  LeRobot:    {lerobot.__version__}')
except ImportError as e:
    print(f'  WARNING: LeRobot not found: {e}')
    print('  To install: uv pip install lerobot')

try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print('  SmolVLA:    OK (lerobot.common.policies.smolvla)')
except ImportError:
    print('  SmolVLA:    not found via LeRobot — will use fallback')

try:
    import peft
    print(f'  PEFT:       {peft.__version__}')
except ImportError:
    print('  WARNING: peft not found')
" || true

# ── 5. Verify ManiSkill ──────────────────────────────────────────────────────
echo "[5/5] Checking ManiSkill3..."
uv run python -c "
try:
    import mani_skill
    print(f'  ManiSkill:  {mani_skill.__version__}')
except ImportError as e:
    print(f'  WARNING: ManiSkill not found: {e}')
    print('  To install: uv pip install mani-skill')
" || true

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Collect demos:        uv run python scripts/collect_demos.py --all-training"
echo "  2. Zero-shot baseline:   uv run python scripts/run_zero_shot.py"
echo "  3. SFT baseline:         uv run python scripts/run_sft.py"
echo "  4. Continual RL (main):  uv run python scripts/run_continual_rl.py"
echo "  5. Full benchmark:       uv run python scripts/benchmark.py"
echo ""
echo "Quick smoke test:"
echo "  uv run python scripts/benchmark.py --dry-run"
