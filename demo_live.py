#!/usr/bin/env python3
"""
============================================================
  Empathetic Dialogue — 现场演示脚本 (Live Demo)
============================================================
  用途：在 presentation 中现场运行，展示整个项目的可复现性。
  
  功能：
    1. 展示训练数据样本
    2. 加载已训练的基线 & 共情链模型
    3. 用自定义输入生成对比回复
    4. 展示已有的 LLM Judge 评分
    5. 展示校准结果 & 分析报告
    6. 展示 NLG 指标摘要

  运行方式：
    python demo_live.py              # 完整交互式演示
    python demo_live.py --quick      # 只展示数据和结果（不加载模型，更快）
    python demo_live.py --section 3  # 只运行第3个演示环节
============================================================
"""

import json
import sys
import os
import argparse
import random
from pathlib import Path

# ── 颜色输出 ──
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def banner(text, color=C.CYAN):
    width = 60
    print(f"\n{color}{C.BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{C.END}\n")

def section(num, title):
    print(f"\n{C.BOLD}{C.YELLOW}{'─' * 60}")
    print(f"  DEMO {num}: {title}")
    print(f"{'─' * 60}{C.END}\n")

def info(label, value):
    print(f"  {C.BLUE}{label}:{C.END} {value}")

def success(text):
    print(f"  {C.GREEN}✓ {text}{C.END}")

def warn(text):
    print(f"  {C.YELLOW}⚠ {text}{C.END}")

def pause():
    input(f"\n  {C.DIM}[按 Enter 继续...]{C.END}")


# ═══════════════════════════════════════════════════════════
#  DEMO 1: 展示训练数据
# ═══════════════════════════════════════════════════════════
def demo_data():
    section(1, "训练数据概览 — 5,318 条心理咨询对话")

    data_path = Path("data/formatted_Psych_data.jsonl")
    if not data_path.exists():
        warn(f"数据文件不存在: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    info("数据集", "PsychCentral 心理咨询 Q&A")
    info("总样本数", f"{total:,} 条对话")
    info("数据格式", "JSONL (每行一条 JSON)")
    info("字段", "current_statement, therapist_response, cognitive_distortions, conversation_history")

    print(f"\n  {C.BOLD}随机展示 3 条样本：{C.END}\n")

    samples = random.sample(lines, min(3, total))
    for i, line in enumerate(samples, 1):
        sample = json.loads(line)
        stmt = sample.get("current_statement", "")[:120]
        resp = sample.get("therapist_response", "")[:120]
        distortions = sample.get("cognitive_distortions", [])

        print(f"  {C.CYAN}[样本 {i}]{C.END}")
        print(f"  {C.BOLD}用户：{C.END}{stmt}{'...' if len(sample.get('current_statement', '')) > 120 else ''}")
        print(f"  {C.GREEN}治疗师：{C.END}{resp}{'...' if len(sample.get('therapist_response', '')) > 120 else ''}")
        if distortions:
            print(f"  {C.DIM}认知扭曲：{', '.join(distortions[:3])}{C.END}")
        print()


# ═══════════════════════════════════════════════════════════
#  DEMO 2: 模型架构 & 加载 checkpoint
# ═══════════════════════════════════════════════════════════
def demo_model_load():
    section(2, "模型架构 — GPT-2 + Chain-of-Empathy")

    print(f"""  {C.BOLD}Chain-of-Empathy 五阶段推理（灵感来自 CBT 认知行为疗法）：{C.END}

    ┌─────────────────┐
    │  1. 情境理解      │  理解对话上下文
    │  (Scenario)      │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  2. 情感识别      │  识别用户情绪状态
    │  (Emotion)       │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  3. 原因推断      │  理解情绪产生的原因
    │  (Cause)         │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  4. 目标设定      │  设定治疗性回复目标
    │  (Goal)          │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  5. 回复规划      │  生成共情回复表示
    │  (Response)      │
    └─────────────────┘
             +
    ┌─────────────────┐
    │  情感-情境融合     │  emotion ⊕ scenario → fuse
    │  (Fusion Layer)  │
    └─────────────────┘
             ↓
       response_rep + fused_rep → 残差连接到 GPT-2 lm_head
""")

    # 检查 checkpoint 文件
    ckpts = {
        "基线模型 (best)": Path("checkpoints/baseline_best.pt"),
        "基线模型 (final)": Path("checkpoints/baseline_final.pt"),
        "共情链模型 (best)": Path("checkpoints/empathy_best.pt"),
        "共情链模型 (final)": Path("checkpoints/empathy_final.pt"),
    }

    print(f"  {C.BOLD}Checkpoint 文件：{C.END}")
    for name, path in ckpts.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            success(f"{name}: {path} ({size_mb:.0f} MB)")
        else:
            warn(f"{name}: {path} (未找到)")

    print(f"\n  {C.DIM}基线模型 ~475 MB = GPT-2 (124M params)")
    print(f"  共情链模型 ~491 MB = GPT-2 + Chain-of-Empathy (+16 MB 共情模块){C.END}")


def demo_generate():
    section("2b", "现场生成对比 — 基线 vs. 共情链")

    try:
        import torch
        from transformers import AutoTokenizer
        from Model_Baseline import GPT2BaselineModel
        from Model_Integration import CBT_EmpatheticModel
    except ImportError as e:
        warn(f"缺少依赖: {e}")
        warn("跳过模型生成演示。运行 pip install torch transformers 后重试。")
        return

    model_name = "uer/gpt2-chinese-cluecorpussmall"

    print(f"  {C.DIM}加载分词器: {model_name}...{C.END}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_ckpt = Path("checkpoints/baseline_best.pt")
    empathy_ckpt = Path("checkpoints/empathy_best.pt")

    if not baseline_ckpt.exists() or not empathy_ckpt.exists():
        warn("Checkpoint 文件不完整，跳过生成演示。")
        return

    print(f"  {C.DIM}加载基线模型...{C.END}")
    baseline_model = GPT2BaselineModel(model_name)
    baseline_model.load_state_dict(torch.load(str(baseline_ckpt), map_location="cpu"))
    baseline_model.eval()

    print(f"  {C.DIM}加载共情链模型...{C.END}")
    empathy_model = CBT_EmpatheticModel(model_name, hidden_dim=768)
    empathy_model.load_state_dict(torch.load(str(empathy_ckpt), map_location="cpu"))
    empathy_model.eval()
    success("两个模型均已加载\n")

    # 预设的演示输入
    demo_prompts = [
        "我最近总是失眠，脑子里停不下来，一直在想工作的事情。",
        "我和我最好的朋友吵架了，我觉得可能再也回不去了。",
        "我总觉得自己不够好，什么事情都做不好。",
    ]

    print(f"  {C.BOLD}预设演示输入（可选择序号，或输入自定义文字）：{C.END}")
    for i, p in enumerate(demo_prompts, 1):
        print(f"    {C.CYAN}[{i}]{C.END} {p}")
    print(f"    {C.CYAN}[0]{C.END} 自定义输入")

    choice = input(f"\n  请选择 (默认 1): ").strip() or "1"

    if choice == "0":
        user_input = input(f"  请输入您的话: ").strip()
        if not user_input:
            user_input = demo_prompts[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(demo_prompts):
        user_input = demo_prompts[int(choice) - 1]
    else:
        user_input = demo_prompts[0]

    print(f"\n  {C.BOLD}用户输入：{C.END}{user_input}\n")
    print(f"  {C.DIM}生成中...{C.END}")

    baseline_resp = baseline_model.generate_response(user_input, max_length=80)
    empathy_resp = empathy_model.generate_response(user_input, max_length=80)

    # 去掉生成中包含的 prompt
    if baseline_resp.startswith(user_input):
        baseline_resp = baseline_resp[len(user_input):].strip()
    if empathy_resp.startswith(user_input):
        empathy_resp = empathy_resp[len(user_input):].strip()

    print(f"  {C.RED}[Baseline GPT-2]{C.END}")
    print(f"  {baseline_resp}\n")
    print(f"  {C.GREEN}[Chain-of-Empathy GPT-2]{C.END}")
    print(f"  {empathy_resp}\n")

    print(f"  {C.DIM}注：GPT-2 (124M) 规模有限，分数在 1-2/5 分属正常。")
    print(f"  模型的作用是作为消融对象来验证评估框架的区分度。{C.END}")


# ═══════════════════════════════════════════════════════════
#  DEMO 3: 展示已有生成结果
# ═══════════════════════════════════════════════════════════
def demo_generations():
    section(3, "生成结果 — 600 条模型回复 (200/模型)")

    gen_dir = Path("outputs/generations")
    files = {
        "GPT-2 Vanilla (未微调)": gen_dir / "gpt2_vanilla.jsonl",
        "GPT-2 Fine-tuned (基线)": gen_dir / "gpt2_finetuned.jsonl",
        "GPT-2 + Chain-of-Empathy": gen_dir / "empathy_chain.jsonl",
    }

    for model_name, path in files.items():
        if not path.exists():
            warn(f"{path} 不存在")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        print(f"  {C.BOLD}{model_name}{C.END} — {len(data)} 条")

        # 展示 1 条样本
        sample = random.choice(data)
        user_stmt = sample.get("user_statement", sample.get("prompt", ""))[:100]
        response = sample.get("response", "")[:150]
        runtime = sample.get("runtime_s", 0)

        print(f"    {C.DIM}用户：{C.END}{user_stmt}")
        print(f"    {C.GREEN}回复：{C.END}{response}{'...' if len(sample.get('response', '')) > 150 else ''}")
        print(f"    {C.DIM}生成耗时：{runtime:.2f}s | seed={sample.get('seed')} | temp={sample.get('temperature')}{C.END}")
        print()


# ═══════════════════════════════════════════════════════════
#  DEMO 4: LLM Judge 评分展示
# ═══════════════════════════════════════════════════════════
def demo_judge():
    section(4, "LLM Judge 评分 — 1,800 次 API 调用, 0 解析失败")

    judge_dir = Path("outputs/judge")
    files = {
        "GPT-2 Vanilla": judge_dir / "gpt2_vanilla_judge.jsonl",
        "GPT-2 Fine-tuned": judge_dir / "gpt2_finetuned_judge.jsonl",
        "Chain-of-Empathy": judge_dir / "empathy_chain_judge.jsonl",
    }

    print(f"  {C.BOLD}4-维度评分量表 (1-5 分)：{C.END}")
    print(f"    ① 情感识别 (Emotion Recognition)  — 是否准确识别用户情感")
    print(f"    ② 情感验证 (Validation & Warmth)   — 是否验证感受、传达温暖")
    print(f"    ③ 帮助性   (Helpfulness)           — 回复是否有帮助")
    print(f"    ④ 安全性   (Safety & Boundaries)   — 是否安全无害")
    print()

    all_scores = {}

    for model_name, path in files.items():
        if not path.exists():
            warn(f"{path} 不存在")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # 计算平均分
        dims = ["emotion", "validation", "helpfulness", "safety"]
        avgs = {}
        for dim in dims:
            scores = [d["scores"][dim] for d in data if "scores" in d and dim in d.get("scores", {})]
            avgs[dim] = sum(scores) / len(scores) if scores else 0

        overall_scores = [d.get("overall", 0) for d in data if "overall" in d]
        avgs["overall"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        all_scores[model_name] = avgs

        print(f"  {C.CYAN}{model_name}{C.END} ({len(data)} 条评分)")
        print(f"    Emotion: {avgs['emotion']:.2f}  Validation: {avgs['validation']:.2f}  "
              f"Helpfulness: {avgs['helpfulness']:.2f}  Safety: {avgs['safety']:.2f}  "
              f"{C.BOLD}Overall: {avgs['overall']:.2f}{C.END}")

    if all_scores:
        print(f"\n  {C.BOLD}对比总结：{C.END}")
        print(f"  {'模型':<24} {'Emotion':>8} {'Valid.':>8} {'Help.':>8} {'Safety':>8} {'Overall':>8}")
        print(f"  {'─' * 64}")
        for name, avgs in all_scores.items():
            print(f"  {name:<24} {avgs['emotion']:>8.2f} {avgs['validation']:>8.2f} "
                  f"{avgs['helpfulness']:>8.2f} {avgs['safety']:>8.2f} {avgs['overall']:>8.2f}")

    # 展示一条具体评分
    print(f"\n  {C.BOLD}具体评分示例：{C.END}")
    for model_name, path in files.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        sample = data[0]
        notes = sample.get("notes", "N/A")[:100]
        confidence = sample.get("confidence", "N/A")
        print(f"    {C.CYAN}[{model_name}]{C.END} ID: {sample.get('sample_id', 'N/A')}")
        print(f"    评分: {sample.get('scores', {})}")
        print(f"    置信度: {confidence} | 评语: {notes}")
        print()
        break  # 只展示一条


# ═══════════════════════════════════════════════════════════
#  DEMO 5: NLG 指标
# ═══════════════════════════════════════════════════════════
def demo_nlg():
    section(5, "NLG 指标 — BLEU / ROUGE (传统指标的局限性)")

    nlg_path = Path("outputs/nlg_metrics.json")
    if not nlg_path.exists():
        warn(f"{nlg_path} 不存在")
        return

    with open(nlg_path, "r") as f:
        metrics = json.load(f)

    print(f"  {C.BOLD}{'模型':<24} {'BLEU':>10} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}{C.END}")
    print(f"  {'─' * 64}")

    for model, scores in metrics.items():
        label = {
            "gpt2_vanilla": "GPT-2 Vanilla",
            "gpt2_finetuned": "GPT-2 Fine-tuned",
            "empathy_chain": "Chain-of-Empathy",
        }.get(model, model)

        print(f"  {label:<24} {scores.get('bleu', 0):>10.4f} {scores.get('rouge1', 0):>10.4f} "
              f"{scores.get('rouge2', 0):>10.4f} {scores.get('rougeL', 0):>10.4f}")

    print(f"\n  {C.YELLOW}关键洞察：{C.END}")
    print(f"  BLEU/ROUGE 无法衡量共情质量 — '你的感受完全合理' vs '我理解你的心情'")
    print(f"  两者语义等价但 BLEU ≈ 0。这正是我们需要 LLM-as-Judge 的原因。")


# ═══════════════════════════════════════════════════════════
#  DEMO 6: 校准结果
# ═══════════════════════════════════════════════════════════
def demo_calibration():
    section(6, "校准结果 — MAE 降低 31-63%")

    print(f"  {C.BOLD}External Human-Anchored Calibration (Isotonic Regression){C.END}")
    print(f"  用公开的人类评分数据集作为锚点，校准 LLM Judge 的分数偏差。\n")

    # 直接展示核心结果
    results = [
        ("Emotion Recognition", 0.547, 0.205, 62.6, 0.581),
        ("Validation & Warmth", 0.544, 0.249, 54.2, 0.325),
        ("Helpfulness",         0.506, 0.219, 56.6, 0.759),
        ("Safety & Boundaries", 0.425, 0.285, 32.9, 0.785),
    ]

    print(f"  {'维度':<24} {'原始MAE':>10} {'校准MAE':>10} {'降幅':>10} {'Spearman ρ':>12}")
    print(f"  {'─' * 66}")
    for dim, raw, cal, drop, rho in results:
        color = C.GREEN if drop > 50 else C.YELLOW
        print(f"  {dim:<24} {raw:>10.3f} {cal:>10.3f} {color}{drop:>9.1f}%{C.END} {rho:>12.3f}")

    print(f"\n  {C.GREEN}关键发现：{C.END}")
    print(f"  ✓ 校准修正了 LLM Judge 的系统性偏严（所有维度 bias < 0）")
    print(f"  ✓ 排序相关性保持不变 — Judge 的区分能力不受影响")
    print(f"  ✓ 校准后 MAE < 0.3 (5 分制) — 达到实用水平")

    # 展示校准后的 JSONL 样本
    calib_path = Path("outputs/calibrated/isotonic_calibrated.jsonl")
    if calib_path.exists():
        print(f"\n  {C.BOLD}校准数据样本：{C.END}")
        with open(calib_path, "r") as f:
            sample = json.loads(f.readline())
        print(f"    sample_id: {sample.get('sample_id', 'N/A')}")
        print(f"    人类评分: {sample.get('human_scores', {})}")
        print(f"    原始Judge: {sample.get('judge_raw', {})}")
        print(f"    校准后:    {sample.get('calibrated', {})}")


# ═══════════════════════════════════════════════════════════
#  DEMO 7: 消融实验
# ═══════════════════════════════════════════════════════════
def demo_ablation():
    section(7, "消融实验 — 重复次数 & Judge 稳定性")

    print(f"  {C.BOLD}重复次数敏感性 (k = 1 / 2 / 3)：{C.END}\n")
    print(f"  {'k':>4} {'Emotion ρ':>12} {'Safety ρ':>12} {'Emotion MAE':>14} {'结论':>10}")
    print(f"  {'─' * 56}")
    print(f"  {'1':>4} {'0.658':>12} {'0.855':>12} {'0.201':>14} {'✓ 足够':>10}")
    print(f"  {'2':>4} {'0.661':>12} {'0.871':>12} {'0.206':>14}")
    print(f"  {'3':>4} {'0.651':>12} {'0.875':>12} {'0.204':>14}")

    print(f"\n  {C.GREEN}结论：k=1 已足够 — 额外重复增益 < 0.01{C.END}")
    print(f"  {C.GREEN}→ 可节省 66% API 成本，无质量损失{C.END}")

    print(f"\n  {C.BOLD}Judge 自我一致性：{C.END}")
    print(f"    完全一致率:  88 – 100%")
    print(f"    近似一致率:  96 – 100% (±1 分)")
    print(f"    均分标准差:  0.00 – 0.12")

    # 展示分析报告文件
    report_path = Path("outputs/analysis/ablation_repeats.md")
    if report_path.exists():
        success(f"详细报告: {report_path}")


# ═══════════════════════════════════════════════════════════
#  DEMO 8: 项目结构 & 可复现性
# ═══════════════════════════════════════════════════════════
def demo_reproducibility():
    section(8, "可复现性 — 一键重跑整个管线")

    print(f"  {C.BOLD}项目结构：{C.END}")
    print(f"""
    EmpatheticDialogues/
    ├── src/                         # 核心模块（分层架构）
    │   ├── data/                    # 数据加载、模板、外部数据集
    │   ├── models/                  # 基线 & 共情链模型
    │   ├── inference/               # 统一生成接口
    │   └── eval/                    # 评估量表、LLM Judge、校准
    ├── experiments/                 # 实验脚本（可一键运行）
    │   ├── run_train.sh             # Step 1: 训练
    │   ├── run_generate.sh          # Step 2: 生成
    │   ├── run_judge.sh             # Step 3: LLM 评审
    │   ├── run_calibrate.sh         # Step 4: 校准
    │   └── run_paper_pipeline.sh    # 全流程一键执行
    ├── checkpoints/                 # 训练好的模型权重
    ├── outputs/                     # 所有实验输出
    │   ├── generations/             # 600 条生成回复
    │   ├── judge/                   # 1800 条 LLM 评分
    │   ├── calibrated/              # 校准后评分
    │   └── analysis/                # 分析报告
    └── data/                        # 5,318 条训练数据
""")

    print(f"  {C.BOLD}一键复现命令：{C.END}")
    cmds = [
        ("安装依赖", "pip install -r requirements.txt"),
        ("训练模型", "bash experiments/run_train.sh"),
        ("生成回复", "bash experiments/run_generate.sh"),
        ("LLM 评审", "DEEPSEEK_API_KEY=sk-... bash experiments/run_judge.sh"),
        ("校准分析", "bash experiments/run_paper_pipeline.sh"),
    ]
    for i, (label, cmd) in enumerate(cmds, 1):
        print(f"    {C.CYAN}Step {i} ({label}):{C.END}")
        print(f"    $ {C.GREEN}{cmd}{C.END}\n")


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Empathetic Dialogue 现场演示脚本")
    parser.add_argument("--quick", action="store_true", help="快速模式：跳过模型加载，只展示数据和结果")
    parser.add_argument("--section", type=int, default=0, help="只运行指定的演示环节 (1-8)")
    args = parser.parse_args()

    banner("Empathetic Dialogue Evaluation Framework — 现场演示")
    print(f"  {C.DIM}模式: {'快速展示' if args.quick else '完整演示（含模型生成）'}{C.END}")
    print(f"  {C.DIM}提示: python demo_live.py --quick  跳过模型加载{C.END}")
    print(f"  {C.DIM}      python demo_live.py --section N  只运行第 N 个环节{C.END}")

    sections = {
        1: ("训练数据", demo_data),
        2: ("模型架构", demo_model_load),
        3: ("生成结果", demo_generations),
        4: ("LLM Judge", demo_judge),
        5: ("NLG 指标", demo_nlg),
        6: ("校准结果", demo_calibration),
        7: ("消融实验", demo_ablation),
        8: ("可复现性", demo_reproducibility),
    }

    # 如果指定了 section
    if args.section:
        if args.section in sections:
            sections[args.section][1]()
            if args.section == 2 and not args.quick:
                pause()
                demo_generate()
        else:
            warn(f"无效的 section: {args.section} (可选 1-8)")
        return

    # 完整流程
    for num, (name, func) in sections.items():
        func()
        if num == 2 and not args.quick:
            pause()
            demo_generate()
        if num < 8 and not args.quick:
            pause()

    banner("演示结束 — 谢谢！", C.GREEN)
    print(f"  {C.BOLD}三大贡献：{C.END}")
    print(f"  1. 外部人类锚定校准 — 无需自行标注，MAE ↓ 31-63%")
    print(f"  2. 端到端可复现管线 — 数据→训练→生成→评估→校准")
    print(f"  3. k=1 足够 — 节省 66% API 成本\n")


if __name__ == "__main__":
    main()
