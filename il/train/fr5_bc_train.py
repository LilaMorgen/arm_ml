import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from il.env.fr5_pybullet_env_no_block import FR5_Env


# 定义仿真环境（示例：使用PyBullet的Kuka机械臂）
def make_env(gui:bool=False):
    env = FR5_Env(gui=gui)
    env.reset()
    return env


# 专家策略（示例：随机动作）
def expert_policy(state):
    # return np.random.uniform(-1, 1, size=(6,))  # 6个关节
    model = PPO.load(r"../../models/fr5_pybullet/PPO/model94.zip")
    action, _ = model.predict(observation=state, deterministic=True)
    return action


# 数据收集
def collect_expert_data(num_episodes):
    env = make_env(True)
    data = []
    for _ in range(num_episodes):
        env.reset()
        state = env.get_observation()
        done = False
        while not done:
            action = expert_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            data.append((state, action))
            state = next_state
            done = terminated or truncated

    env.close()
    return data


# 数据预处理
def preprocess_data(expert_data):
    states = np.array([np.array(d[0]).flatten() for d in expert_data])
    actions = np.array([np.array(d[1]).flatten() for d in expert_data])
    state_scaler = StandardScaler()
    states_normalized = state_scaler.fit_transform(states)
    X_train, X_val, y_train, y_val = train_test_split(
        states_normalized, actions, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val, state_scaler


# 模型定义
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 训练函数
def train_model(X_train, y_train, X_val, y_val, input_dim, output_dim, n_epoch):
    model = BCModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t)
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    return model


# 主流程
if __name__ == "__main__":
    # 收集数据
    # expert_data = collect_expert_data(100)
    # with open("../gen_data/expert_data.pkl", "wb") as f:
    #     pickle.dump(expert_data, f)
    # 加载生成的数据
    data = pickle.load(open("../gen_data/expert_data.pkl", "rb"))
    print(f"states data shape: {np.array(data[0][0]).flatten().shape}")
    print(f"actions data shape: {np.array(data[0][1]).shape}")

    # 预处理数据
    X_train, X_val, y_train, y_val, scaler = preprocess_data(data)

    # 训练模型
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = train_model(X_train, y_train, X_val, y_val, input_dim, output_dim, 500)

    # 保存模型
    torch.save(model.state_dict(), "../../models/fr5_pybullet/BC/bc_model.pth")