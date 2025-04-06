import torch.nn as nn

def initialize_chain_of_empathy(model):
    # 假设你使用的是全连接层，使用 Xavier 初始化方法初始化权重
    nn.init.xavier_uniform_(model.chain.scenario_layer.weight)
    nn.init.xavier_uniform_(model.chain.emotion_layer.weight)
    nn.init.xavier_uniform_(model.chain.cause_layer.weight)
    nn.init.xavier_uniform_(model.chain.goal_layer.weight)
    nn.init.xavier_uniform_(model.chain.response_layer.weight)

    # 初始化偏置项
    nn.init.zeros_(model.chain.scenario_layer.bias)
    nn.init.zeros_(model.chain.emotion_layer.bias)
    nn.init.zeros_(model.chain.cause_layer.bias)
    nn.init.zeros_(model.chain.goal_layer.bias)
    nn.init.zeros_(model.chain.response_layer.bias)

    return model
