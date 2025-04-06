import torch
import torch.nn as nn
import torch.nn.functional as F

class ChainOfEmpathy(nn.Module):
    def __init__(self, hidden_dim):
        super(ChainOfEmpathy, self).__init__()
        # 定义各步骤对应的线性层
        self.scenario_layer = nn.Linear(hidden_dim, hidden_dim)  # 情境理解
        self.emotion_layer = nn.Linear(hidden_dim, hidden_dim)   # 情感识别
        self.cause_layer = nn.Linear(hidden_dim, hidden_dim)     # 原因推断
        self.goal_layer = nn.Linear(hidden_dim, hidden_dim)      # 目标设定
        self.response_layer = nn.Linear(hidden_dim, hidden_dim)  # 回复生成

        # 可以考虑增加多个层次来更好地捕捉情感信息
        self.emotion_fusion_layer = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x):
        # 情境理解
        scenario_rep = F.relu(self.scenario_layer(x))
        # 提取情感信息
        emotion_rep = F.relu(self.emotion_layer(scenario_rep))
        # 分析情感产生的原因
        cause_rep = F.relu(self.cause_layer(emotion_rep))
        # 根据分析确定应对目标
        goal_rep = F.relu(self.goal_layer(cause_rep))
        # 生成回复的特征表示
        response_rep = self.response_layer(goal_rep)

        # 情感和情境信息的融合
        fused_rep = torch.cat((emotion_rep, scenario_rep), dim=-1)
        fused_rep = F.relu(self.emotion_fusion_layer(fused_rep))

        # 可以在最终返回时加入融合后的表示来增强回复的共情性
        return response_rep + fused_rep
