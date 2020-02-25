import matplotlib.pyplot as plt
# plt.use('pdf')
import numpy as np 
from pylab import *
import pandas as pd
from scipy import signal
f = plt.figure()
# plt.rcParams.update({'font.size': 35})
# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=28)
# plt.rc('ytick', labelsize=28)
# plt.rc('axes', labelsize=28)

# ticks = np.linspace(-10, 100, num=12)
# plt.yticks(ticks)

data = pd.read_csv('./frame_1/run_.,tag_true_positives.csv')
y = data['Value']
# y = signal.savgol_filter(y, 21, 2) # window size 51, polynomial order 3
y = y[0:500]
z = -0.001
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.plot(z)
plt.plot(y, linewidth=3, color='b')
plt.grid()
# plt.show()
f.savefig("true_positives.png", bbox_inches='tight')
