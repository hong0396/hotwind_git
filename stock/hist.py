import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
 
 
# style set 这里只是一些简单的style设置
# sns.set_palette('deep', desat=.6)
# sns.set_context(rc={'figure.figsize': (8, 5) } )
np.random.seed(1425)
data = randn(10)
print(data)

plt.hist(data)
plt.show()



import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
 
 
# example data
mu = 100 # mean of distribution
sigma = 15 # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)
 
num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
 
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()


import seaborn as sns
sns.set(style="ticks")
exercise = sns.load_dataset("exercise")
print(exercise)