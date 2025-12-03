# Installation Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd counterfactual-reasoning
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

**Activate it:**
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Package in Editable Mode
```bash
pip install -e .
```

### 5. Install SMAC Maps
Follow the instructions in [`SETUP_SMAC.md`](SETUP_SMAC.md) to download and install the StarCraft II maps.

---

## Verify Installation

### Test FrozenLake
```bash
cd examples
jupyter notebook frozenlake_demo.ipynb
```

### Test SMAC
```bash
cd examples
jupyter notebook smac_demo.ipynb
```

---

## Training an Agent

### Train PPO on SMAC
```bash
python train_ppo_smac.py
```

This will train for ~2-4 hours and save checkpoints to `models/ppo_smac_3m/`.

---

## Troubleshooting

**Import errors**: Make sure you ran `pip install -e .`

**SMAC errors**: Verify StarCraft II is installed at `C:\Program Files (x86)\StarCraft II` (Windows) or `~/StarCraftII` (Mac/Linux)

**Missing maps**: Download from the [SMAC repository](https://github.com/oxwhirl/smac) and extract to `Maps/SMAC_Maps/`
