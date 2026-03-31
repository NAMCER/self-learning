import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 绘制标准光源 D65 的光谱功率分布
# D65 = 标准日光，色温 6500K，是最常用的白点参考

wavelengths = np.arange(380, 781, 5)  # nm

# 简化的 Planck 黑体辐射公式 (色温 6500K)
T = 6500
h, c, k = 6.626e-34, 3e8, 1.381e-23
lam = wavelengths * 1e-9
spd = (2*h*c**2 / lam**5) / (np.exp(h*c/(lam*k*T)) - 1)
spd /= spd.max()

plt.figure(figsize=(10, 4))
plt.plot(wavelengths, spd, 'gold', linewidth=2)
plt.fill_between(wavelengths, spd, alpha=0.3, color='yellow')
plt.xlabel(f'波长 (nm)')
plt.ylabel(f'相对功率')
plt.title(f'D65 标准光源光谱功率分布 (6500K)')
plt.grid(alpha=0.3)
plt.show()