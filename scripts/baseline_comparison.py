import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ========================================
# لود دیتا
# ========================================

df = pd.read_csv("Dataset/SemEval15/Test/Restaurants_Test.csv")

# استخراج نمونه ها
samples = []
for _, row in df.iterrows():
    text = row['raw_text']
    aspects = eval(row['aspectTerms'])
    for asp in aspects:
        term = asp['term']
        polarity = asp['polarity'].lower()
        if term != 'noaspectterm' and polarity != 'none':
            samples.append({
                'text': text,
                'aspect': term,
                'polarity': polarity
            })

print(f"Total samples: {len(samples)}")

# شمارش هر کلاس
pos_count = sum(1 for s in samples if s['polarity'] == 'positive')
neg_count = sum(1 for s in samples if s['polarity'] == 'negative')
neu_count = sum(1 for s in samples if s['polarity'] == 'neutral')

print(f"\nClass distribution:")
print(f"  Positive: {pos_count} ({pos_count/len(samples)*100:.1f}%)")
print(f"  Negative: {neg_count} ({neg_count/len(samples)*100:.1f}%)")
print(f"  Neutral:  {neu_count} ({neu_count/len(samples)*100:.1f}%)")

# majority class 
majority_class = 'positive' if pos_count >= neg_count and pos_count >= neu_count else \
                 'negative' if neg_count >= neu_count else 'neutral'
print(f"  Majority class: {majority_class}")

# ========================================
# Baseline 1: Random
# ========================================

print(f"\n{'='*50}")
print("Baseline 1: Random")
print(f"{'='*50}")

random.seed(42)  # برای تکرارپذیری
random_correct = 0
for s in samples:
    pred = random.choice(['positive', 'negative', 'neutral'])
    if pred == s['polarity']:
        random_correct += 1

random_acc = random_correct / len(samples) * 100
print(f"→ Accuracy: {random_acc:.2f}%")

# ========================================
# Baseline 2: Majority
# ========================================

print(f"\n{'='*50}")
print(f"Baseline 2: Majority (always '{majority_class}')")
print(f"{'='*50}")

majority_correct = sum(1 for s in samples if s['polarity'] == majority_class)
majority_acc = majority_correct / len(samples) * 100
print(f"→ Accuracy: {majority_acc:.2f}%")

# ========================================
# Model: InstructABSA
# ========================================

print(f"\n{'='*50}")
print("Model: InstructABSA")
print(f"{'='*50}")

model_name = "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

instruction = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

model_correct = 0
for s in tqdm(samples):
    prompt = instruction.format(text=s['text'], aspect=s['aspect'])
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)
    
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    if pred == s['polarity']:
        model_correct += 1

model_acc = model_correct / len(samples) * 100
print(f"→ Accuracy: {model_acc:.2f}%")

# ========================================
# نتایج نهایی
# ========================================

print(f"\n{'='*60}")
print("📊 Final Comparison: Baselines vs InstructABSA")
print(f"{'='*60}")
print(f"{'Method':<30} {'Accuracy':<15} {'Bar'}")
print(f"{'-'*60}")

results = {
    'Random Baseline': random_acc,
    f'Majority Baseline ({majority_class})': majority_acc,
    'InstructABSA (Paper)': 84.50,
    'InstructABSA (Ours)': model_acc,
}

for name, acc in results.items():
    bar = "█" * int(acc / 5)
    print(f"{name:<30} {acc:5.2f}%        {bar}")

# بهبود نسبت به baseline
print(f"\n{'='*60}")
print("📈 Improvement over Baselines")
print(f"{'='*60}")
print(f"  vs Random:   +{model_acc - random_acc:.2f}%")
print(f"  vs Majority: +{model_acc - majority_acc:.2f}%")

# ذخیره
results_df = pd.DataFrame({
    'Method': list(results.keys()),
    'Accuracy': list(results.values())
})
results_df.to_csv('Output/baseline_comparison.csv', index=False)
print(f"\n📁 Saved to Output/baseline_comparison.csv")
