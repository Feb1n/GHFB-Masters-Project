from Linear_Fit import read_data
from Linear_Fit import exp_decay_fit
import matplotlib.pyplot as plt
import numpy as np

files = ['B3S2','B3S4','B3S6','B3S8']
filetype = '.csv'

data = []

def error_calc(dataset):
    ratio = np.array(dataset[1])   #ratio of signal/reference intensity
    signal = np.array(dataset[3]) #signal intensity raw data
    ref = np.array(dataset[4])    #reference intensity raw data
    signaldev = np.array(dataset[5]) #std dev of repeat measurements for signal raw data
    refdev = np.array(dataset[6]) #std dev of repeat measurements for reference raw data

    # Propagate errors in regions to ratio
    ratiodev = ratio * np.sqrt((signaldev / signal)**2 + (refdev / ref)**2)

    log_ratiodev = ratiodev / (ratio * np.log(10))

    return log_ratiodev


for file in files: data.append(read_data(file+filetype))


for i in data: plt.errorbar(np.log10(i[0]),np.log10(i[1]),
                            yerr=error_calc(i),
                            marker='o',
                            label = files[data.index(i)],
                            linestyle = 'none',
                            elinewidth=1,   
                            capsize=3)

plt.title('Log graph of the ratio of counts per second for samples in Batch 3')
plt.xlabel('Log(Ratio/Second)')
plt.ylabel('Log(Current)')
plt.legend()
plt.show()
