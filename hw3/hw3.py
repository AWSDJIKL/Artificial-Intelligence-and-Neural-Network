import numpy as np


# 感知器
def my_perceptron(x_list, y_list, max_t):
    w = np.array([0, 0, 0, 0]).reshape(4, 1)
    t = 0
    while t < max_t:
        for x, y in zip(x_list, y_list):
            # fx = np.dot(w.T, x)
            isFalse = w.T.dot(x * y) <= 0
            if isFalse:
                w += x * y
            t += 1
            if t >= max_t:
                break
        if t >= max_t:
            break
    return w


# 计算正确率
def classify(w, x_list, y_list):
    correct = 0
    for x, y in zip(x_list, y_list):
        # 结果同号，说明预测正确
        if np.dot(w.T, x)[0][0] * y > 0:
            correct += 1
    return correct / len(x_list)


# 导入数据
data_set = np.loadtxt('data_x1x2x3y.csv', dtype=np.int64, delimiter=',', skiprows=1)
x_list = []
y_list = []
for data in data_set:
    x_list.append(np.r_[data[0:3], [1]].reshape(4, 1))
    y_list.append(data[3])
# 分批测试不同迭代次数对正确率的影响
for t in [200, 50, 20, 5]:
    w = my_perceptron(x_list[0:80], y_list[0:80], t)
    print("T={}，权重向量w={}，正确率为：{}".format(t, w, classify(w, x_list[-80:], y_list[-80:])))
