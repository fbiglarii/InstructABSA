import pandas as pd #Reading CSV files and working with data tables
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #loading model and tokenizer from HuggingFace
import torch # PyTorch for model execution
from tqdm import tqdm # for showing progress bar 

#  بررسي سخت افزار و تنظيمات سيستم
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# لود مدل
model_name = "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded!")

# لود دیتا
df = pd.read_csv("Dataset/SemEval15/Test/Restaurants_Test.csv")
print(f"Loaded {len(df)} samples")

# Instruction
instruction = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

# ارزیابی
correct = 0
total = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row['raw_text'] # جمله
    aspects = eval(row['aspectTerms']) #Aspect list
    
    for asp in aspects:
        term = asp['term']  #  aspect name
        true_polarity = asp['polarity'].lower()  #true sentiment
        
        if term == 'noaspectterm' or true_polarity == 'none':
            continue
        
        prompt = instruction.format(text=text, aspect=term)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        if pred == true_polarity:
            correct += 1
        total += 1

# نتیجه
accuracy = correct / total * 100
print(f"\n{'='*40}")
print(f"Results on SemEval15 Restaurants Test")
print(f"{'='*40}")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"\nPaper reports: 84.50%")
