import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Generator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto"
        )

    def generate(self, query, context, max_tokens=100):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Prompt entfernen
        answer = output_text.split("Answer:")[-1].strip()

        return answer
    
    def generate_ground_truth(query):
        prompt = f"Provide a concise, factual answer:\n{query}"
    
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
