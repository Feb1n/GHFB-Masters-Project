import pandas 
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


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


def xy_graph(data):
    '''
    Plots given data as a basic xy graph, without taking the log of the axes.

    Parameters
    ----------
    data : list of list
        2D list in the form [x, y].

    Returns
    -------
    None
    '''
    plt.plot(data[0], data[1], marker='.', label='Measured Points')
    r_2 = r_squared(data)
    
    plt.title('Standard Graph')
    plt.xlabel('Current (mA)')
    plt.ylabel('Counts Per Second')
    plt.grid(True)
    plt.annotate(f'r² = {r_2:.4f}', xy=(min(data[0]), 0.9*max(data[1])), fontsize=10)
    plt.legend()
    plt.show()


def log_graph(data):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.
    Includes error bars based on standard deviation of both regions across repeat measurement.
    
    Parameters:
    

    Returns:
    Logarithmic graph with error bars
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
    ratiodev = ratio * np.sqrt((signaldev / signal)**2 + (refdev / ref)**2)

    # Propagate error
    log_ratiodev = ratiodev / (ratio * np.log(10))
  
    # Take the log of the current and counts
    data[0] = np.log10(current)
    data[1] = np.log10(ratio)

    # Plot the data with error bars in black
    plt.errorbar(
        data[0], 
        data[1], 
        yerr=log_ratiodev, 
        fmt='.', 
        color='black',   # Colour of the data points
        ecolor='black',  # Colour of the error bars
        elinewidth=1,    # Width of the error bar lines
        capsize=3        # Size of the error bar caps
    )

    # Linear fit (optional, for visualization)
    linear_fit(data)
    
    # Calculate and display R-squared value
    r_2 = r_squared(data)
    print(f'R^2: {r_2}')
    
    # Display plot details
    plt.title('Logarithmic Graph with Error Bars')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Second')
    plt.grid(True)
    plt.annotate(f'R^2 = {r_2:.3f}', xy=(min(data[0]), 0.9 * max(data[1])))
    plt.legend()
    plt.show()




def r_squared(data):
    '''
    Calculates the R² score for a linear fit to the data.

    Parameters
    ----------
    data : list of list
        2D list in the form [x, y].

    Returns
    -------
    r_squared : float
        The R² score of the fit.
    '''
    x = np.array(data[0])
    y = np.array(data[1])
    coef, intercept = np.polyfit(x, y, 1)
    predicted = coef * x + intercept
    r_squared = r2_score(y, predicted)

    return r_squared


def linear_fit(data):
    '''
    Performs a linear fit and plots the fit line.

    Parameters
    ----------
    data : list of list
        2D list in the form [x, y].

    Returns
    -------
    None
    '''
    coef = np.polyfit(data[0], data[1], 1)
    poly1d_fn = np.poly1d(coef)
    
    plt.plot(data[0], poly1d_fn(data[0]), label='Linearity Guide')


file = 'B2S6.csv'
data1 = read_data(file)
log_graph(data1)
