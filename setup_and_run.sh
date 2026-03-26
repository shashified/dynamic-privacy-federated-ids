#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  DynamicPrivacyIDS — One-shot setup + run
#  Usage:  bash setup_and_run.sh
# ─────────────────────────────────────────────────────────────────────────
set -e

echo "════════════════════════════════════════════"
echo "  DynamicPrivacyIDS Setup"
echo "════════════════════════════════════════════"

# 1. Install dependencies
pip install -q "flwr[torch]>=1.8" torch opacus \
    pandas matplotlib seaborn numpy scikit-learn \
    jupyterlab ipykernel 2>&1 | tail -5

echo "✅  Dependencies installed"

# 2. (Optional) Clone base repo for real FLNET2023 data
if [ ! -d "FML-Network" ]; then
    echo "Cloning base repo..."
    git clone --depth 1 https://github.com/nsol-nmsu/FML-Network.git
    # Link data folder if present
    if [ -d "FML-Network/data" ]; then
        ln -sf FML-Network/data data
        echo "✅  Linked FLNET2023 data"
    fi
fi

# 3. Run main simulation
echo ""
echo "────────────────────────────────────────────"
echo "  Running DynamicPrivacyIDS simulations ..."
echo "  (3 modes × 3 seeds = 9 runs, ~5-10 min)"
echo "────────────────────────────────────────────"
python dynamic_fl_ids.py

echo ""
echo "════════════════════════════════════════════"
echo "  ✅  Done!  Results in results/"
echo "  📒  Launch notebook:"
echo "      jupyter notebook experiments.ipynb"
echo "════════════════════════════════════════════"
