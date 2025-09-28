import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# 生成两个随机分布作为示例
P = np.random.rand(64, 50, 768)
Q = np.random.rand(64, 50, 768)

# 对每个 batch 进行平均处理
P_mean = np.mean(P, axis=0)
Q_mean = np.mean(Q, axis=0)

# 计算每个特征维度的JS散度
js_divergences = [jensenshannon(P_mean[:, i], Q_mean[:, i]) for i in range(P_mean.shape[1])]

# 绘制JS散度
plt.figure(figsize=(10, 6))
plt.plot(js_divergences, label='JS Divergence')
plt.xlabel('Feature Dimension')
plt.ylabel('JS Divergence')
plt.title('JS Divergence Across Feature Dimensions')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# 生成两个随机分布作为示例
P = np.random.rand(64, 50, 768)
Q = np.random.rand(64, 50, 768)
P1 = np.random.rand(64, 50, 768)
Q1 = np.random.rand(64, 50, 768)
# 对每个 batch 进行平均处理
P_mean = np.mean(P, axis=0)
Q_mean = np.mean(Q, axis=0)
P_mean1 = np.mean(P, axis=0)
Q_mean1 = np.mean(Q, axis=0)
# 计算每个特征维度的JS散度
js_divergences = [jensenshannon(P_mean[:, i], Q_mean[:, i]) for i in range(P_mean.shape[1])]
js_divergences1 = [jensenshannon(P_mean1[:, i], Q_mean1[:, i]) for i in range(P_mean1.shape[1])]
# 绘制JS散度
plt.figure(figsize=(10, 6))
plt.plot(js_divergences, label='JS Divergence', color='red')
plt.plot(js_divergences1, label='JS Divergence1', color='blue')
plt.xlabel('Feature Dimension')
plt.ylabel('JS Divergence')
plt.title('JS Divergence Across Feature Dimensions')
plt.legend()
plt.show()