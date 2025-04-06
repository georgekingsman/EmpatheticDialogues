import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Chain_of_Empathy import ChainOfEmpathy
from Model_Integration import CBT_EmpatheticModel


class CBT_EmpatheticModel(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.device = torch.device("cpu")  # 默认为CPU，如果有GPU可改为cuda
        self.model_name = model_name
        # 加载 GPT-2 模型
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 初始化共情链模块
        self.chain = ChainOfEmpathy(hidden_dim)

    def forward(self, input_ids, attention_mask):
        # GPT-2 模型的输出
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态
        pooled = hidden_states.mean(dim=1)  # 平均池化，得到一个 (batch_size, hidden_dim) 的表示

        # 通过共情链模块生成情感表示
        chain_out = self.chain(pooled)
        # 使用加权拼接的方式结合情感表示与 GPT-2 的输出

        chain_expanded = chain_out.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # 扩展为与输入相同的长度

        # 加权拼接情感嵌入与 GPT-2 的隐藏状态
        combined_hidden = torch.cat([hidden_states, chain_expanded], dim=-1)  # 拼接隐藏状态和情感表示
        logits = self.gpt2.lm_head(combined_hidden)  # 生成 logit 作为输出
        return logits

    def generate_response(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            # 获取 GPT-2 嵌入
            input_embeds = self.gpt2.get_input_embeddings()(input_ids)
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)  # 平均池化得到情感表示

            # 通过共情链模块生成情感表示
            chain_out = self.chain(pooled)
            chain_expanded = chain_out.unsqueeze(1).repeat(1, input_embeds.size(1), 1)

            weight = 0.5  # 可以调整权重，以平衡情感嵌入与GPT-2输出的影响
            # 加权结合情感嵌入与 GPT-2 嵌入
            conditioned_embeds = input_embeds + weight * chain_expanded  # 调整权重

            # 使用 GPT-2 的生成方法生成文本
            generated = self.gpt2.generate(
                inputs_embeds=conditioned_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)



def generate_response_with_baseline(model, tokenizer, prompt, max_length=50):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 使用生成模型生成回复
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

    # 解码生成的ID
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def generate_response_with_empathy(model, tokenizer, prompt, max_length=50):
    # 生成共情链模型的回复
    empathy_response = model.generate_response(prompt, max_length=max_length)
    return empathy_response


def main():
    model_name = "gpt2"  # 这里可以换成你使用的其他模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 使用 CPU
    device = torch.device("cpu")

    # 载入基准模型和共情链模型
    model_baseline = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model_empathy = CBT_EmpatheticModel(model_name, hidden_dim=768).to(device)  # 使用你的训练模型

    prompt = "我总觉得自己不够好，什么事情都做不好。"

    # 生成基准模型的回复
    baseline_response = generate_response_with_baseline(model_baseline, tokenizer, prompt)
    print("基准模型回复：", baseline_response)

    # 生成共情链模型的回复
    empathy_response = generate_response_with_empathy(model_empathy, tokenizer, prompt)
    print("共情链模型回复：", empathy_response)

    # 可以通过人工或其他评价标准进行对比
    print("\n对比实验结果：")
    print("基准模型回复与共情链模型回复进行对比：")
    print("基准模型回复：", baseline_response)
    print("共情链模型回复：", empathy_response)


if __name__ == "__main__":
    main()
