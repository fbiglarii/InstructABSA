# InstructABSA - Reproduction and Improvement

This project reproduces and improves results from the paper "InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis" (NAACL 2024).

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kevinscaria/InstructABSA.git
cd InstructABSA
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
InstructABSA/
├── Demo/         	   # Jupyter notebook and sample data
├── Docs/         	   # Model outputs and charts
├── scripts/      	   # Our evaluation scripts
└── requirements.txt   # Dependencies
```

## Our Scripts

| Script | Description |
|--------|-------------|
| `scripts/simple_eval.py` | Reproduces ATSC task results on Rest15 dataset |
| `scripts/Reproduce_ATSC_results.py` | Reproduces ATSC results on all datasets |
| `scripts/inference.py` | Inference on custom input file |
| `scripts/Prompt_engineering.py` | Tests 4 and 8 examples in prompt |
| `scripts/test_flan_t5.py` | Compares Flan-T5 with Tk-Instruct |
| `scripts/baseline_comparison.py` | Compares with Random and Majority baselines |
| `scripts/persian_comparison.py` | Cross-lingual test on Persian data |

## Usage

### Run evaluation
```bash
python scripts/simple_eval.py
```

### Run inference on custom data
```bash
python scripts/inference.py --input demo/sample_input.csv --output Output/predictions.csv
```

## Results

| Experiment | Accuracy |
|------------|----------|
| Paper (baseline) | 84.50% |
| Ours (4 examples) | 85.98% (+1.48%) |
| Flan-T5 (zero-shot) | 84.69% |

## References

- Paper: [InstructABSA](https://arxiv.org/abs/2302.08624)
- GitHub: [kevinscaria/InstructABSA](https://github.com/kevinscaria/InstructABSA)
- HuggingFace: [kevinscaria](https://huggingface.co/kevinscaria)

## Hardware

- CPU: Intel (no GPU required)
- RAM: 8GB+
- OS: Windows 10/11



## requirements.txt:
```
torch
transformers
pandas
tqdm
matplotlib
sentencepiece
```
