import json
import random
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from Model_Baseline import GPT2BaselineModel
from Model_Integration import CBT_EmpatheticModel
import evaluate


class TherapistQADataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_text": sample["current_statement"],
            "target_text": sample["therapist_response"]
        }


def collate_fn(batch, tokenizer, max_length=128):
    input_texts = [item["input_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs["input_ids"], inputs["attention_mask"], targets["input_ids"]


def generate_response(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



def main():
    # 模型名称
    model_name = "uer/gpt2-chinese-cluecorpussmall"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载基线模型
    baseline_model = GPT2BaselineModel(model_name)
    baseline_model.load_state_dict(torch.load('./gpt2_baseline.pt', map_location=torch.device('cpu')))
    baseline_model.eval()

    # 加载共情链增强模型
    empathy_model = CBT_EmpatheticModel(model_name, hidden_dim=768)
    empathy_model.load_state_dict(torch.load('./cbt_gpt2_model.pt', map_location=torch.device('cpu')))
    empathy_model.eval()

    # 加载数据集
    dataset = TherapistQADataset("./data/formatted_Psych_data.jsonl")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    # 获取一个批次的测试样本
    input_ids, attention_mask, target_ids = next(iter(dataloader))

    # 解码输入和目标文本
    input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    reference_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]

    # 存储模型生成的回复
    baseline_outputs = []
    empathy_outputs = []

    # 对每个测试样本生成回复
    for input_text in input_texts:
        # 基线模型生成回复
        baseline_response = generate_response(baseline_model, tokenizer, input_text)
        baseline_outputs.append(baseline_response)

        # 共情链增强模型生成回复
        empathy_response = generate_response(empathy_model, tokenizer, input_text)
        empathy_outputs.append(empathy_response)

        # 打印结果
        print(f"用户输入: {input_text}")
        print(f"参考回复: {reference_texts[input_texts.index(input_text)]}")
        print(f"基线模型回复: {baseline_response}")
        print(f"共情链增强模型回复: {empathy_response}")
        print("-" * 80)

    # 加载评估指标
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    # 计算 BLEU 分数
    baseline_bleu = bleu.compute(predictions=baseline_outputs, references=[[ref] for ref in reference_texts])
    empathy_bleu = bleu.compute(predictions=empathy_outputs, references=[[ref] for ref in reference_texts])

    # 计算 ROUGE 分数
    baseline_rouge = rouge.compute(predictions=baseline_outputs, references=reference_texts)
    empathy_rouge = rouge.compute(predictions=empathy_outputs, references=reference_texts)

    # 打印评估结果
    print("基线模型 BLEU 得分:", baseline_bleu["bleu"])
    print("共情链增强模型 BLEU 得分:", empathy_bleu["bleu"])
    print("基线模型 ROUGE 得分:", baseline_rouge)
    print("共情链增强模型 ROUGE 得分:", empathy_rouge)


if __name__ == "__main__":
    main()
