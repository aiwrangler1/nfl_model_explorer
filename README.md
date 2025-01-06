# NFL 2024 EDP Model Exploration

This project implements an Expected Drive Points (EDP) model for NFL analysis, incorporating strength of schedule adjustments and advanced metrics.

## Project Structure

```
nfl-2024-edp-model-exploration/
├── edp_analysis/           # Main package directory
│   ├── analysis_interface/ # Analysis and visualization components
│   ├── core_calculation/   # Core EDP calculation logic
│   ├── data_management/    # Data loading and configuration
│   └── utils/             # Utility functions and helpers
│
├── data/                  # Data storage
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data files
│
├── tests/                # Test suite
│   ├── analysis/        # Analysis component tests
│   ├── data/           # Data processing tests
│   └── processing/     # Pipeline tests
│
├── docs/                # Documentation
├── logs/               # Application logs
└── model_outputs/      # Model outputs and visualizations
```

## Setup

1. Create a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main analysis script can be run with:
```bash
python -m edp_analysis.analysis_interface.edp_rankings
```

For more detailed documentation, see `docs/PROJECT.md`.
