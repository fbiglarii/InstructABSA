import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ========================================
# لود دیتا (کل Rest15)
# ========================================

df = pd.read_csv("Dataset/SemEval15/Test/Restaurants_Test.csv")

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

print(f"Testing on ALL {len(samples)} samples")

# ========================================
# Prompts
# ========================================

# Zero-shot
prompt_zero_shot = """What is the sentiment toward the aspect in the review? Answer with one word: positive, negative, or neutral.

Review: {text}. Aspect: {aspect}. Sentiment:"""

# 6-shot
prompt_6_shot = """What is the sentiment toward the aspect in the review? Answer with: positive, negative, or neutral.

Example 1:
Review: The food was delicious. Aspect: food. Sentiment: positive

Example 2:
Review: I loved the atmosphere. Aspect: atmosphere. Sentiment: positive

Example 3:
Review: The service was terrible. Aspect: service. Sentiment: negative

Example 4:
Review: The wait time was too long. Aspect: wait time. Sentiment: negative

Example 5:
Review: The price is reasonable. Aspect: price. Sentiment: neutral

Example 6:
Review: They serve Italian food. Aspect: food. Sentiment: neutral

Now answer:
Review: {text}. Aspect: {aspect}. Sentiment:"""

# ========================================
# لود Flan-T5
# ========================================

print("\n" + "="*50)
print("Loading Flan-T5-base...")
print("="*50)

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded!")

# ========================================
# تابع ارزیابی
# ========================================

def evaluate(prompt_template, samples):
    correct = 0
    
    for s in tqdm(samples):
        prompt = prompt_template.format(text=s['text'], aspect=s['aspect'])
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        if 'positive' in pred:
            pred_label = 'positive'
        elif 'negative' in pred:
            pred_label = 'negative'
        elif 'neutral' in pred:
            pred_label = 'neutral'
        else:
            pred_label = pred
        
        if pred_label == s['polarity']:
            correct += 1
    
    return correct / len(samples) * 100

# ========================================
# تست هر دو حالت
# ========================================

print("\n" + "="*50)
print("Test 1: Flan-T5 Zero-shot")
print("="*50)
zero_shot_acc = evaluate(prompt_zero_shot, samples)
print(f"→ Accuracy: {zero_shot_acc:.2f}%")

print("\n" + "="*50)
print("Test 2: Flan-T5 6-shot")
print("="*50)
six_shot_acc = evaluate(prompt_6_shot, samples)
print(f"→ Accuracy: {six_shot_acc:.2f}%")

# ========================================
# نتایج نهایی
# ========================================

print(f"\n{'='*65}")
print(f"📊 Full Comparison on Rest15 ({len(samples)} samples)")
print(f"{'='*65}")
print(f"{'Model':<40} {'Accuracy':<15}")
print(f"{'-'*65}")
print(f"{'Random Baseline':<40} {'33.03%':<15}")
print(f"{'Majority Baseline':<40} {'60.15%':<15}")
print(f"{'Flan-T5-base (zero-shot)':<40} {zero_shot_acc:.2f}%")
print(f"{'Flan-T5-base (6-shot)':<40} {six_shot_acc:.2f}%")
print(f"{'InstructABSA (6-shot, fine-tuned)':<40} {'84.50%':<15}")
print(f"{'='*65}")


# ذخیره
results_df = pd.DataFrame({
    'Model': ['Random', 'Majority', 'Flan-T5 (0-shot)', 'Flan-T5 (6-shot)', 'InstructABSA'],
    'Accuracy': [33.03, 60.15, zero_shot_acc, six_shot_acc, 84.50]
})
results_df.to_csv('Output/flan_t5_comparison.csv', index=False)
print(f"\n📁 Saved to Output/flan_t5_comparison.csv")
