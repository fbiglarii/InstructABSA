"""
اسکریپت Inference برای فایل ورودی جدید
استفاده: python inference.py --input my_data.csv --output results.csv

فرمت فایل ورودی:
text,aspect
"The food was great",food
"Service was slow",service
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
from tqdm import tqdm

def main(input_file, output_file):
    # تنظیم device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # لود مدل
    model_name = "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded!")
    
    #  prompt (با 4 مثال)
    instruction = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Example 1-
input: The food was delicious. The aspect is food.
output: positive

Example 2-
input: The service was terrible. The aspect is service.
output: negative

Example 3-
input: The price is reasonable. The aspect is price.
output: neutral

Example 4-
input: I loved the atmosphere. The aspect is atmosphere.
output: positive

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""
    
    # لود داده
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")
    
    # چک کردن ستونهای مورد نیاز
    if 'text' not in df.columns or 'aspect' not in df.columns:
        print("Error: Input file must have 'text' and 'aspect' columns!")
        return
    
    results = []
    
    # پیش بینی
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row['text']
        aspect = row['aspect']
        
        prompt = instruction.format(text=text, aspect=aspect)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        results.append({
            'text': text,
            'aspect': aspect,
            'predicted_sentiment': pred
        })
    
    # ذخیره نتایج
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # نمایش چند نمونه
    print(f"\nSample predictions:")
    print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aspect-Based Sentiment Classification Inference')
    parser.add_argument('--input', required=True, help='Input CSV file with text and aspect columns')
    parser.add_argument('--output', required=True, help='Output CSV file for predictions')
    args = parser.parse_args()
    
    main(args.input, args.output)
