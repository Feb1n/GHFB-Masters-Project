from Linear_Fit import read_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Specify csv filenames to be plotted on one graph in the list below:
files = ['B4S4 620BP','B4S5']

def log_graph(data,file):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.
    Includes error bars based on standard deviation of both regions across repeat measurement.
    
    Parameters:
    
    1 dataset to be plotted in a 1D array.
    Name of the sample
    Returns:
    Logarithmic graph with error bars
    Logarithmic plot with error bars and a linear fit.
    '''

    # Extract parameters from the data
    current = np.array(data[0]) # current values
    ratio = np.array(data[1])   #ratio of signal/reference intensity
    inttime = np.array(data[2]) #variable integration time
    signal = np.array(data[3]) #signal intensity raw data
    ref = np.array(data[4])    #reference intensity raw data
    signaldev = np.array(data[5]) #std dev of repeat measurements for signal raw data
    refdev = np.array(data[6]) #std dev of repeat measurements for reference raw data

   # Propagate errors in regions to ratio
    ratiodev = ratio * inttime*1e-3* np.sqrt((signaldev / signal)**2 + (refdev / ref)**2)

    # Propagate error
    log_ratiodev = ratiodev / (ratio * np.log(10))

    # Take the log of the current and counts
    x_data = np.log10(current)
    y_data = np.log10(ratio)

    # Plot the data with error bars in black
    plt.errorbar(
        x_data, 
        y_data, 
        yerr=log_ratiodev, 
        fmt='.', 
        color='black',   # Colour of the data points
        ecolor='black',  # Colour of the error bars
        elinewidth=1,    # Width of the error bar lines
        capsize=3        # Size of the error bar caps
    )

    # Linear fit (optional, for visualization)
    popt_lin, popc_lin = curve_fit(lin_fit, x_data, y_data,  sigma = log_ratiodev, absolute_sigma=True)
    lin_chi_squared = calc_chi_squared(y_data, x_data, log_ratiodev, lin_fit, *popt_lin)
    lin_red_chi_squared = calc_red_chi_squared(lin_chi_squared, x_data, len(popt_lin))
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    plt.plot(x_fit, lin_fit(x_fit, *popt_lin), label=file +"  m = "+str(round(popt_lin[0],2))+' ±' +str(round(np.sqrt(popc_lin[0,0]),2))+'  '+r"$\bar{\chi}^2$ = %.3f" % lin_red_chi_squared)  # Hide duplicate legend entry


    # Display plot details
    plt.title('Logarithmic Graph with Error Bars')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Second')
    plt.grid(True)
    plt.legend()

def lin_fit(x, m, c):
  return m*x + c

#Defining chi-squared
def calc_chi_squared(y_data, x_data, y_err, fit_model, *params):
  model_values = fit_model(x_data, *params)
  chi_squared = np.sum(((y_data - model_values) / y_err) ** 2)
  return chi_squared

#Defining reduced chi-squared
def calc_red_chi_squared(chi_squared, x_data, N):
  DoF = len(x_data) - N   #N is number of fitting parameters
  red_chi_squared = (1/DoF) * chi_squared
  return red_chi_squared

# Adjust sizing of figure
plt.figure(figsize=(10, 6))

figure_name = 'Multi_plot'
for file in files:
    data = read_data(file+'.csv')
    log_graph(data,file)
    figure_name = figure_name+'_'+file

# Adjust Titles and labels on figure
plt.title('Log graph of the ratio of counts per second')
plt.ylabel('Log(Ratio/Second)')
plt.xlabel('Log(Current)')
plt.savefig('images/'+figure_name+'.png')
plt.show()

