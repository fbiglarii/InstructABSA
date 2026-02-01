# InstructABSA - Reproduction and Improvement

This project reproduces and improves results from the paper "InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis" (NAACL 2024).

## Installation

### 1. Clone the repository
```bash
cd C:\Users\YourUsername\Desktop
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
pip install torch torchvision torchaudio
pip install transformers datasets pandas scikit-learn openpyxl tqdm evaluate accelerate sentencepiece
pip install -r requirements.txt
```

## Project Structure
```
InstructABSA/
├── Dataset/           # SemEval datasets
├── InstructABSA/      # Main model code
├── Notebooks/         # Jupyter Notebooks for training
├── Output/            # Model outputs
├── Scripts/           # Shell scripts
├── instructions.py    # Instruction definitions
├── run_model.py       # Main execution script
└── requirements.txt   # Dependencies
```

## Our Scripts

| Script | Description |
|--------|-------------|
| `simple_eval.py` | Simple model evaluation - reproduces ATSC task results on Rest15 dataset |
| `Reproduce_ATSC_results.py` | Reproduces ATSC task results on all datasets (Rest14, Rest15, Rest16, Lapt14) |
| `inference.py` | Inference on custom user input (ATSC task with text + aspect input). Usage: `python inference.py --input sample_input.csv --output results.csv` |
| `Prompt_engineering.py` | Tests different number of examples (4 and 8) in prompt for ATSC task on Rest15 dataset |
| `test_flan_t5.py` | Compares Flan-T5 model with Tk-Instruct (InstructABSA) |
| `baseline_comparison.py` | Compares model output with Random and Majority baselines |
| `persian_comparison.py` | Evaluates model performance on Persian language (cross-lingual test) |


## References

- Paper: [InstructABSA](https://arxiv.org/abs/2302.08624)
- GitHub: [kevinscaria/InstructABSA](https://github.com/kevinscaria/InstructABSA)
- HuggingFace: [kevinscaria](https://huggingface.co/kevinscaria)

## Hardware

- CPU: Intel (no GPU required)
- RAM: 8GB+
- OS: Windows 10/11

