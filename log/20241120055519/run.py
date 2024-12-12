#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import physo
import physo.learn.monitoring as monitoring
from physo.benchmark.utils import symbolic_utils as su
import sympy
import os
from datetime import datetime
import sys  # 导入sys模块
import logging  # 用于日志记录

# 设置日志配置
def setup_logging(log_dir):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(log_dir, f"{timestamp}_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_file

# 日志记录函数
def log_message(message):
    logging.info(message)
    print(message)  # 同时打印到终端，方便调试

# 数据加载
data = pd.read_csv("/home/linux1/PhySO/all-r11193.csv")

# 定义特征的名称，即 x
features_name = ['d', 'f', 'e1', 'e2', 'u1', 'u2']

# 量纲元素的数目与特征的数目保持一致
features_name_units = [[0], [0], [0], [0], [0], [0]]

# 定义标签的名称，即y
label_name = "r"

# 定义标签的量纲
label_name_units = [0]

# 定义自由常数的名称
free_consts_names = ['a', 'b']  # 用实际的常数名称替换

# 格式化输入的数据
msg = [list(data[feature_name]) for feature_name in features_name]
X = np.stack(tuple(msg), axis=0)
y = np.array(data[label_name])

# 保存文件的函数
def save_files_with_timestamp(data_path, log_dir):
    try:
        # 初始化日志
        log_file = setup_logging(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = os.path.join(log_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        log_message(f"Created directory: {save_dir}")  # 记录到日志

        # 数据点绘图
        n_dim = X.shape[0]
        fig, ax = plt.subplots(n_dim, 1, figsize=(10, 5))
        for i in range(n_dim):
            curr_ax = ax if n_dim == 1 else ax[i]
            curr_ax.plot(X[i], y, 'k.',)
            curr_ax.set_xlabel(f"X[{i}]")
            curr_ax.set_ylabel("y")
        plt.savefig(os.path.join(save_dir, 'output.png'))
        plt.close(fig)  # 关闭图形，避免在显示中重复绘制
        log_message(f"Figure saved in: {os.path.join(save_dir, 'output.png')}")  # 记录到日志

        # 保存日志和可视化
        save_path_training_curves = os.path.join(save_dir, 'demo_curves.png')
        save_path_log = os.path.join(save_dir, 'demo.log')

        run_logger = lambda: monitoring.RunLogger(save_path=save_path_log, do_save=True)
        run_visualiser = lambda: monitoring.RunVisualiser(epoch_refresh_rate=1,
                                                          save_path=save_path_training_curves,
                                                          do_show=False,
                                                          do_prints=True,
                                                          do_save=True)

        return run_logger, run_visualiser, save_dir

    except Exception as e:
        log_message(f"An error occurred: {e}")  # 记录异常信息
        raise  # 抛出异常，防止函数继续执行

start_time = datetime.now()

try:
    # 运行SR任务
    run_logger, run_visualiser, save_dir = save_files_with_timestamp(
        data_path="/home/linux1/PhySO/all-r11193.csv", log_dir="/home/linux1/PhySO/log")
    
    expression, logs = physo.SR(
        X, y,
        X_names=features_name,
        X_units=features_name_units,
        y_name=label_name,
        y_units=label_name_units,
        fixed_consts=[0.02093333333],
        fixed_consts_units=[[0]],
        free_consts_names=["a", "b"],
        free_consts_units=[[0], [0]],
        op_names=["add", "sub", "mul", "sub", "exp", "log", "tanh"],
        get_run_logger=run_logger,
        get_run_visualiser=run_visualiser,
        run_config=physo.config.config0.config0,
        parallel_mode=False,
        epochs=500
    )

    best_expr = expression

    log_message(f"\n最佳表达式ascii ：\n\n{best_expr.get_infix_pretty()}")
    log_message(f"\n最佳表达式latex ：\n{best_expr.get_infix_latex()}")
    log_message(f"\n带系数的最佳表达式:\n\n{best_expr.get_infix_sympy(evaluate_consts=True)[0]}")
    log_message(f"\n带系数的最佳表达式latex:\n\n{sympy.latex(best_expr.get_infix_sympy(evaluate_consts=True))}")
    log_message(f"\n自由系数维度:\n\n{best_expr.free_consts}")
    log_message(f"\n自由系数:\n\n{best_expr.free_consts.class_values}")

except Exception as e:
    log_message(f"An error occurred during SR process: {e}")

# 记录程序结束时间
end_time = datetime.now()
elapsed_time = end_time - start_time
log_message(f"\n程序总耗时：\n\n{elapsed_time}")
