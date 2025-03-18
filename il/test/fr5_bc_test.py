import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from il.env.fr5_pybullet_env_no_block import FR5_Env
from il.core.bc_algorithm import BCModel

# 定义仿真环境（示例：使用PyBullet的Kuka机械臂）
def make_env(gui:bool=False):
    env = FR5_Env(gui=gui)
    env.reset()
    return env

# 主流程
if __name__ == "__main__":
    policy = BCModel(12,6)
    policy.load_state_dict(torch.load("../../models/fr5_pybullet/BC/bc_model.pth"))
    policy.eval()

    env = make_env(True)
    env.render()

    test_num = 100  # 测试次数
    success_num = 0  # 成功次数
    with torch.no_grad():
        for i in range(test_num):
            next_state, _ = env.reset()
            done = False
            score = 0
            while not done:
                action = policy(torch.tensor(np.array(next_state).flatten(), dtype=torch.float32))
                next_state, reward, done, _, info = env.step(action)
                score += reward
            if info['is_success']:
                success_num += 1
            print("奖励：", score)
        success_rate = success_num / test_num
        print("成功率：", success_rate)
        env.close()