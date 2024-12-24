'''
Calculates boundary effects
'''
import numpy as np
import seaborn as sns
import pandas as pd
from pylab import *

## Dataset
n = 10000
noise_amplitude = .5
mean_z = np.random.uniform(0, 1., n)
z = mean_z + noise_amplitude*(np.random.rand(n)-.5)
df = pd.DataFrame({'z':z, 'mean_z':mean_z})

## Calculate E[mean_z | z]
sns.lmplot(data=df, x='z', y='mean_z', lowess=True, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red', 'linewidth': 2})

show()
