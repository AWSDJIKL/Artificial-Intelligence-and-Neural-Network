import numpy as np
import matplotlib.pyplot as plt


# 原函数
def y(x):
    result = pow(((5 + 4 * np.cos(x)) / 4), 0.5) + pow(((5 - 4 * np.cos(x)) / 4), 0.5)
    return result


# 原函数求导，用于求梯度
def dy(x):
    result = 0.5 * np.sin(x) * (pow(((5 - 4 * np.cos(x)) / 4), -0.5) - pow(((5 + 4 * np.cos(x)) / 4), -0.5))
    return result


# 第一部分
n = 100  # 迭代次数
a = 0.2  # 学习效率
for i in range(1, 5):
    x = i
    x_list = [x]
    for j in range(n):
        x -= a * dy(x)
        x_list.append(x)
    print(x_list)
    subpicture = plt.subplot(2, 2, i)
    subpicture.set_title("x0={},x100={},y100={}".format(i, x, y(x)))
    plt.plot(range(101), x_list, label="x")
    plt.plot(range(101), y(x_list), label="y")
    plt.legend()
plt.suptitle("第一部分：alpha=0.2，迭代次数t=100", fontproperties="KaiTi", fontsize=30)
plt.show()
# 第二部分
n = 100  # 迭代次数
a_list = [0.2, 0.5, 1, 5]  # 学习效率
for i in range(1, 5):
    x = 1
    x_list = [x]
    a = a_list[i - 1]
    for j in range(n):
        x += a * dy(x)
        x_list.append(x)
    print(x_list)
    subpicture = plt.subplot(2, 2, i)
    subpicture.set_title("alpha={},x100={},y100={}".format(a, x, y(x)))
    plt.plot(range(101), x_list, label="x")
    plt.plot(range(101), y(x_list), label="y")
    plt.legend()
plt.suptitle("第二部分：x0=1，迭代次数t=100", fontproperties="KaiTi", fontsize=30)
plt.show()

# 显示函数整体曲线，方便观察收敛值是否正常
plt.plot(np.arange(-5, 5, 0.001), y(np.arange(-5, 5, 0.001)))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
