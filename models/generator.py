import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Generator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

    @staticmethod
    def build_context_prompt(query, context):
        return f"""
            You are a question-answering assistant.

            Answer the question using only the information provided in the context.
            Provide a concise but complete factual answer in 2-4 sentences.
            Do not use outside knowledge.
            Do not invent facts.
            If the context does not contain enough information to answer the question completely, don't say so and instead provide the best answer you can with the help of your own knowledge.

            Return only the final answer text.
            Do not repeat the question.
            Do not include labels such as "Question:", "Context:", or "Answer:".

            Context:
            {context}

            Question:
            {query}

            Final answer:
            """.strip()

    @staticmethod
    def build_no_context_prompt(query):
        return f"""
            You are a question-answering assistant.

            Answer the question as best as you can based only on your own knowledge.
            Provide a concise but complete factual answer in 2-4 sentences.
            Do not invent facts.

            Return only the final answer text.
            Do not repeat the question.
            Do not include labels such as "Question:" or "Answer:".

            Question:
            {query}

            Final answer:
            """.strip()

    def _generate_from_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            min_new_tokens=20,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Falls das Modell trotzdem Labels ausgibt
        for label in ["Final answer:", "Answer:", "Question:", "Context:"]:
            answer = answer.replace(label, "").strip()

        return answer

    def generate_with_context(self, query, context):
        prompt = self.build_context_prompt(query, context)
        return self._generate_from_prompt(prompt)

    def generate_without_context(self, query):
        prompt = self.build_no_context_prompt(query)
        return self._generate_from_prompt(prompt)

    @staticmethod
    def generate_ground_truth(query, context):
        prompt = Generator.build_context_prompt(query, context)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return res.choices[0].message.content.strip()