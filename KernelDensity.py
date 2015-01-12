# Reading a new csv file with no headers

import pandas as pd
distanceLabeledToUnlableed = pd.read_csv('/tmp/labeledConfidentialToUnlabeled.csv',',',header=None)

# create a histogram from a one-dimensional data
import numpy as np
from pylab import plot,show,hist
data = distanceLabeledToUnlableed[2].values
histo = np.histogram(np.transpose(data))

# Kernel Density Estimation in  python
from scipy.stats.kde import gaussian_kde
data = distanceLabeledToUnlableed[2]
my_pdf = gaussian_kde(data)
X_plot = np.linspace(np.min(data), np.max(data),1000)
plot(X_plot, my_pdf(X_plot),'r')
show()
