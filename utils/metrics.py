import numpy as np
from sklearn.metrics import r2_score

<<<<<<< HEAD

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def nRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / (np.max(true) - np.min(true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def New_MAPE(true, pred):
    # 防止除以零的情况，给真实值中的零加上一个很小的数
    epsilon = 1e-10
    true = np.array(true)
    pred = np.array(pred)

    # 计算绝对百分比误差
    mape = np.abs((true - pred) / (true + epsilon)) * 100

    # 返回所有样本、时间步、特征的平均MAPE
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


=======
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def nRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / (np.max(true) - np.min(true))

>>>>>>> update
def New_R2(true, pred):
    # 初始化一个列表来存储每个样本的R2值
    r2_scores = []

    # 遍历每个样本（假设 true 和 pred 的形状是一样的）
    for i in range(true.shape[0]):
        # 展平每个样本的序列长度和特征维度
        true_flat = true[i].reshape(-1)
        pred_flat = pred[i].reshape(-1)

        # 计算每个样本的R2
        r2 = r2_score(true_flat, pred_flat)
        r2_scores.append(r2)

    # 返回所有样本的平均R2
    return np.mean(r2_scores)

<<<<<<< HEAD
def r2_score_3d(true, pred):
    # 确保 true 和 pred 都是三维数组
    if true.shape != pred.shape:
        raise ValueError("true and pred must have the same shape")

    # 获取样本数、序列长度和特征数
    num_samples, seq_len, num_features = true.shape

    # 存储每个样本的 R² 值
    r2_scores = []

    # 对每个样本计算 R² 值
    for i in range(num_samples):
        # 取出单个样本的真实值和预测值
        true_sample = true[i]
        pred_sample = pred[i]

        # 计算 R² 值
        r2 = r2_score(true_sample, pred_sample, multioutput='uniform_average')
        r2_scores.append(abs(r2))

    # 计算所有样本的 R² 值的平均
    mean_r2 = np.mean(r2_scores)
    return mean_r2
=======
>>>>>>> update

def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
<<<<<<< HEAD
    rmse = RMSE(pred, true)
    nrmse = nRMSE(pred, true)
    mape = MAPE(pred, true)
    # new_mape = New_MAPE(true,pred)
    mspe = MSPE(pred, true)
    R2   = r2_score_3d(true,pred)
    new_R2 = New_R2(true,pred)


    return mae, mse, rmse, mape,R2 ,mspe,nrmse,new_R2
=======
    nrmse = nRMSE(pred, true)
    new_R2 = New_R2(true,pred)


    return mae, mse,nrmse,new_R2
>>>>>>> update
