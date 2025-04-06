# model_integration.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from Chain_of_Empathy import ChainOfEmpathy

class CBT_EmpatheticModel(nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super(CBT_EmpatheticModel, self).__init__()
        self.device = torch.device("cpu")  # 可根据需要改为 "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.chain = ChainOfEmpathy(hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
        pooled = hidden_states.mean(dim=1)  # 平均池化

        chain_out = self.chain(pooled)  # 经过共情链
        chain_expanded = chain_out.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # 扩展维度
        combined_hidden = hidden_states + chain_expanded  # 融合情感嵌入

        logits = self.gpt2.lm_head(combined_hidden)  # 直接送入语言建模头
        return logits

    def generate(self, **kwargs):
        """
        适配 huggingface 的 generate 接口，支持 inputs_embeds 或 input_ids 调用。
        例如：
            model.generate(input_ids=..., attention_mask=...)
            或
            model.generate(inputs_embeds=..., attention_mask=...)
        """
        return self.gpt2.generate(**kwargs)

    def generate_response(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            input_embeds = self.gpt2.get_input_embeddings()(input_ids)

            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)

            chain_out = self.chain(pooled)
            chain_expanded = chain_out.unsqueeze(1).repeat(1, input_embeds.size(1), 1)
            conditioned_embeds = input_embeds + 0.5 * chain_expanded  # 加权融合

            generated = self.gpt2.generate(
                inputs_embeds=conditioned_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
