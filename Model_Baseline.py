# model_baseline.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

class GPT2BaselineModel(nn.Module):
    def __init__(self, model_name):
        super(GPT2BaselineModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def generate_response(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
