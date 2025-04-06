# data_preprocessing.py
import pandas as pd
import re
import nltk
import json

# 下载 NLTK 的 punkt 模型（仅第一次运行时需要）
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# 定义文本清洗函数
def clean_text(text):
    # 如果输入不是字符串，转换为空字符串
    if not isinstance(text, str):
        text = ""
    text = text.lower()  # 转小写
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    text = re.sub(r"[^a-z0-9\s.,!?']", '', text)  # 保留字母、数字和常用标点
    return text

# 定义认知扭曲检测函数（仅作为示例，可以根据需要扩展规则）
def detect_cognitive_distortions(text):
    distortions = []
    # 示例规则：根据文本内容判断
    if "everything is awful" in text or "nothing will ever change" in text:
        distortions.append("灾难化")
    if "i am a failure" in text:
        distortions.append("自我标签化")
    return distortions

def main():
    # 修改 data_path 为你 CSV 文件的正确路径。如果文件与脚本在同一目录下，可直接使用文件名
    data_path = "../data/Psych_data.csv"  # 或者 "../data/Psych_data.csv"
    df = pd.read_csv(data_path)

    # 查看数据集结构，确认 CSV 文件中的列名
    print("数据集前5行：")
    print(df.head())

    # 根据 CSV 文件中的实际列名设置问题和回答的列名（请修改为实际名称）
    question_col = "Question"
    answer_col = "Answer"

    # 对问题和回答进行预处理，处理缺失值
    df[question_col] = df[question_col].fillna("")
    df[answer_col] = df[answer_col].fillna("")
    df['clean_question'] = df[question_col].apply(clean_text)
    df['clean_answer'] = df[answer_col].apply(clean_text)

    # 分词（可选）
    df['question_tokens'] = df['clean_question'].apply(word_tokenize)
    df['answer_tokens'] = df['clean_answer'].apply(word_tokenize)

    # 检测认知扭曲
    df['cognitive_distortions'] = df['clean_question'].apply(detect_cognitive_distortions)

    # 打印部分预处理结果
    print("预处理后的问题和检测到的认知扭曲：")
    print(df[['clean_question', 'cognitive_distortions']].head())

    # 格式化样本，构建标准格式的 JSON Lines 数据，每行一个 JSON 对象
    def format_sample(row):
        return {
            "conversation_history": row.get("conversation_history", ""),  # 若数据中有对话历史，可用此字段
            "current_statement": row["clean_question"],
            "cognitive_distortions": row["cognitive_distortions"],
            "therapist_response": row["clean_answer"]
        }

    formatted_samples = df.apply(format_sample, axis=1).tolist()

    # 保存为 JSONL 文件，修改输出路径为你希望的目录
    output_path = "../data/formatted_Psych_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in formatted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"数据预处理完成，保存为 {output_path}")

if __name__ == "__main__":
    main()
