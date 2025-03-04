import numpy as np
import matplotlib.pyplot as plt

# 高斯分布的参数
mu = 0     # 均值
sigma = 1  # 标准差

# 生成数据点
x = np.linspace(-5, 5, 1000)
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 绘制高斯分布曲线
plt.plot(x, y, label=f'μ={mu}, σ={sigma}')
plt.title('Gaussian Distribution Curve')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
