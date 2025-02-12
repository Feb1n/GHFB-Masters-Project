from Linear_Fit import read_data
from Linear_Fit import log_graph
import matplotlib.pyplot as plt

# Specify csv filenames to be plotted on one graph in the list below:
files = ['B7S1','B7S2','B7S3','B7S4']

# Adjust sizing of figure
plt.figure(figsize=(10, 6))

for file in files:
    data = read_data(file+'.csv')
    log_graph(data,file)

# Adjust Titles and labels on figure
plt.title('Log graph of the ratio of counts per second')
plt.ylabel('Log(Ratio/Second)')
plt.xlabel('Log(Current)')
plt.legend()
plt.show()