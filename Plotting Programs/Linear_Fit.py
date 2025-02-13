import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D


def read_data(fileName):
    '''
    Reads data from a CSV and returns it, ready to be plotted.

    Parameters
    ----------
    fileName : str
        The name of the CSV file containing data.

    Returns
    -------
    data : list of list
        2D list in the form [all currents in mA, all counts per second].
    '''
    path = 'Data/' + fileName

    with open(path, 'r') as file:
        csv_file = csv.reader(file)
        rows = [line for line in csv_file]

    # Convert data from strings into floats
    data = [[float(i) for i in row] for row in rows]
    return data


def xy_graph(data,file):
    '''
    Plots given data as a basic xy graph and plots an exponential fit onto the data.

    Parameters:
    2D list in the form [x,y]

    Returns:
    XY graph with an exponential fit.
    '''
    current = np.array(data[0])
    exp_time = np.array(data[2])
    ratio = np.array(data[1])   #ratio of signal/reference intensity
    signal = np.array(data[3]) #signal intensity raw data
    ref = np.array(data[4])    #reference intensity raw data
    signaldev = np.array(data[5]) #std dev of repeat measurements for signal raw data
    refdev = np.array(data[6]) #std dev of repeat measurements for reference raw data

    # Propagate errors in regions to ratio
    ratiodev = ratio*exp_time*1e-3*np.sqrt((signaldev / signal)**2 + (refdev / ref) **2)


    # plt.errorbar(current, ratio,fmt='.', label='Measured Points', linestyle='None',color = 'blue')
    plt.errorbar(current, ratio, yerr=ratiodev,fmt='.', label='Measured Points', linestyle='None',color='black',ecolor='black',elinewidth=1,capsize=3)
    exp_decay_fit(data)

    plt.title('Graph of the Ratios of Average Intenseties per Second ')
    plt.xlabel('Current (mA)')
    plt.ylabel('Ratio of Counts Per Second')
    plt.grid(True)
    plt.legend()

def linear_fit(data):
    coef = np.polyfit(data[0], data[1], 1)
    poly1d_fn = np.poly1d(coef)
    slope,intercept = np.poly1d(coef)

    plt.plot(data[0], poly1d_fn(data[0]), label='Linear Fit, m = '+str(slope))

#Defining linear fit
def lin_fit(x, m, c):
  return m*x + c

def log_graph(data,file):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.
    Includes error bars based on standard deviation of both regions across repeat measurement.
    
    Parameters:
    1 dataset to be plotted in a 1D array.
    Name of the sample

    Returns:
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
    ratiodev = ratio*inttime*1e-3*np.sqrt((signaldev / signal)**2 + (refdev / ref)**2)

    # Propagate error
    log_ratiodev = ratiodev /  (ratio*np.log(10))
  
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

    # Linear fit function call
    popt_lin, popc_lin = curve_fit(lin_fit, x_data, y_data,  sigma = log_ratiodev, absolute_sigma=True)
    
    # Calculate and display R-squared value
    r_2 = r_squared(data)
    # Chi-squared values for linear fit
    lin_chi_squared = calc_chi_squared(y_data, x_data, log_ratiodev, lin_fit, *popt_lin)
    lin_red_chi_squared = calc_red_chi_squared(lin_chi_squared, x_data, len(popt_lin))
    


    
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)



# Define the gradient and error
    gradient = popt_lin[0]
    gradient_err = np.sqrt(popc_lin[0, 0])  # Uncertainty in gradient

# Define the gradient and error
    gradient = popt_lin[0]
    gradient_err = np.sqrt(popc_lin[0, 0])  # Uncertainty in gradient

# Create custom legend handles
    line_handle = Line2D([0], [0], color='blue', linestyle='-', label=rf"Linear fit: $y = {gradient:.3f} \pm {gradient_err:.3f}$")
    line_handleblank = Line2D([0], [0], color='white', linestyle='-', label = r"$\bar{\chi}^2$ = %.3f" % lin_red_chi_squared)


# Plot the fit line
    plt.plot(x_fit, lin_fit(x_fit, *popt_lin), color="blue", label="_nolegend_")  # Hide duplicate legend entry

# Manually create a two-line legend
    plt.legend(handles=[line_handle, line_handleblank], loc="best", fontsize=10, frameon=True)

# Display plot details
    plt.title('Logarithmic Graph with Error Bars')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Second')
  


def linear_fit(data,file):
    '''
    Calculates a first degree polynomial fit for provided data.
    Parameters:
    Data to be fitted
    Name of the file

    Returns:
    A plot with a linear fit.
    '''
    coef = np.polyfit(data[0], data[1], 1)
    poly1d_fn = np.poly1d(coef)
    slope,intercept = np.poly1d(coef)



    plt.plot(data[0], poly1d_fn(data[0]), label= file+' Linear Fit, m = '+str(round(slope,2)))



def r_squared(data):
    '''
    Calculates the R^2 score for a linear fit to the data.

    Parameters
    ----------
    data

    Returns
    -------
    r_squared : float
        The R^2 score of the fit.
    '''
    x = np.array(data[0])
    y = np.array(data[1])
    coef, intercept = np.polyfit(x, y, 1)
    predicted = coef * x + intercept
    r_squared = r2_score(y, predicted)

    return r_squared

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

def exp_decay_fit(data):
    '''
    Fits the data to the model y = a + b * exp(-x/t), plots the fitted curve, 
    and calculates the R^2 value.

    Parameters:
    2D list in the form [x, y]

    '''

    # Define the model
    def model(x, a, b, t):
        return a + b * np.exp(-x / t)

    # Normalize data
    x_data = np.array(data[0])
    y_data = np.array(data[1])
    x_data_scaled = x_data / max(x_data)
    y_data_scaled = y_data / max(y_data)

    # Initial guess for fitting parameters
    initial_guess = [0.5, 0.5, 0.5]  # a, b, t in scaled space
    bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])  # Ensure t > 0

    try:
        # Fit the model to the data
        popt, pcov = curve_fit(
            model,
            x_data_scaled,
            y_data_scaled,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        a_fit, b_fit, t_fit = popt
        
        # Rescaling parameters to original data
        a_fit_rescaled = a_fit * max(y_data)
        b_fit_rescaled = b_fit * max(y_data)
        t_fit_rescaled = t_fit * max(x_data)

        # Generate predicted y-values for the original x_data
        y_pred_scaled = model(x_data_scaled, a_fit, b_fit, t_fit)
        y_pred = y_pred_scaled * max(y_data)

        # Calculate R^2 value using sklearn's r2_score
        r_squared = r2_score(y_data, y_pred)
        print(f"R^2 value: {r_squared:.3f}")


        ##requires error to calculate##
        #exp_chi_squared = calc_chi_squared(y_data, x_data, REQUIRES ERROR TO WORK, model, *popt)
        #exp_red_chi_squared = calc_red_chi_squared(exp_chi_squared, x_data, len(popt))

        # Generate fitted curve for plotting
        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = model(x_fit / max(x_data), a_fit, b_fit, t_fit) * max(y_data)

        # Plotting the fitted curve 
        plt.errorbar(x_fit, y_fit,label=f'Fit: a={a_fit_rescaled:.2f}, b={b_fit_rescaled:.2f}, t={t_fit_rescaled:.5f}, R^2={r_squared}', color='green')
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
       


# Main execution
# All files currently in the dataset

files = ['B6S2 smaller', 'B6S2 620 small']



# loop to produce log and XY graphs for all datasets at once and save them into an images folder
for file in files:
    data = read_data(file+'.csv')
    plt.figure(figsize=(10, 6))
    xy_graph(data,file) 
    plt.savefig('images/'+file+'_XY.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    log_graph(data,file)
    plt.savefig('images/'+file+'_log.png')
    plt.close()