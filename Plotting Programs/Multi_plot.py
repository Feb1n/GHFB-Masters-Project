from Linear_Fit import read_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Specify csv filenames to be plotted on one graph in the list below:
files = ['B5S1 620 650 3', 'B5S1 620 650 2']

def log_graph(data, file, index):
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

    # Define marker and color lists
    markers = ['o', 's', '^', 'd', 'x', 'v', '*', 'p', 'h']
    colours = plt.cm.Spectral(np.linspace(0, 1, len(files)))  # Generate distinct colors

    # Extract parameters from the data
    current = np.array(data[0])  # Current values
    ratio = np.array(data[1])  # Ratio of signal/reference intensity
    inttime = np.array(data[2])  # Variable integration time
    signal = np.array(data[3])  # Signal intensity raw data
    ref = np.array(data[4])  # Reference intensity raw data
    signaldev = np.array(data[5])  # Standard deviation of repeat measurements for signal raw data
    refdev = np.array(data[6])  # Standard deviation of repeat measurements for reference raw data

    # Propagate errors in regions to ratio
    ratiodev = ratio * inttime * 1e-3 * np.sqrt((signaldev / signal) ** 2 + (refdev / ref) ** 2)

    # Propagate error
    log_ratiodev = ratiodev / (ratio * np.log(10))

    # Take the log of the current and counts
    x_data = np.log10(current)
    y_data = np.log10(ratio)
        # Linear fit (optional, for visualization)
    popt_lin, popc_lin = curve_fit(lin_fit, x_data, y_data, sigma=log_ratiodev, absolute_sigma=True)
    lin_chi_squared = calc_chi_squared(y_data, x_data, log_ratiodev, lin_fit, *popt_lin)
    lin_red_chi_squared = calc_red_chi_squared(lin_chi_squared, x_data, len(popt_lin))
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    
    # Assign unique marker and color
    marker = markers[index % len(markers)]
    colour = colours[index % len(colours)]
    label_text = (f"{file}: m = {popt_lin[0]:.2f} ± {np.sqrt(popc_lin[0,0]):.2f}, " +
                  r"$\bar{\chi}^2$" + f" = {lin_red_chi_squared:.2f}")

    # Plot the data with error bars
    plt.errorbar(
        x_data, y_data, yerr=log_ratiodev, 
        fmt=marker, color=colour, ecolor='black', 
        elinewidth=1, capsize=3, label=label_text
    )

    # Plot linear fit with same color as datapoint
    plt.plot(x_fit, lin_fit(x_fit, *popt_lin), color=colour, alpha = 0.75, label='_nolegend_')
  

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
for index, file in enumerate(files):  # Use enumerate to get both index and filename
    data = read_data(file+'.csv')
    log_graph(data, file, index)  # Pass index to the function
    figure_name = figure_name + '_' + file

# Adjust Titles and labels on figure
plt.title('Log graph of the ratio of counts per second')
plt.ylabel('Log(Ratio/Second)')
plt.xlabel('Log(Current)')
plt.savefig('images/'+figure_name+'.png')
plt.legend()
plt.show()

