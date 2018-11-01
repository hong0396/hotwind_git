# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# # 生成x轴上的数据:从-3到3，总共有50个点
# x = np.linspace(-1, 1, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
# plt.xlim(-1, 2)
# plt.ylim(-1, 3)
# plt.xlabel("I am x")
# plt.ylabel("I am y")
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(0., np.e, 0.01)
# y1 = np.exp(-x)
# y2 = np.log(x)

# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel('Y values for exp(-x)')
# ax1.set_title("Double Y axis")

# ax2 = ax1.twinx()  # this is the important function
# ax2.bar(x, y2, 'r')
# ax2.set_xlim([0, np.e])
# ax2.set_ylabel('Y values for ln(x)')
# ax2.set_xlabel('Same X for both exp(-x) and ln(x)')

# plt.show()
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

mu = 200
sigma = 25
x = np.random.normal(mu, sigma, size=100)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))

ax0.hist(x, 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75)
ax0.set_title('stepfilled')

# Create a histogram by providing the bin edges (unequally spaced).
bins = [100, 150, 180, 195, 205, 220, 250, 300]
ax1.hist(x, bins, density=True, histtype='bar', rwidth=0.8)
ax1.set_title('unequal bins')

fig.tight_layout()
plt.show()