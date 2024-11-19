import pandas
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


def read_data(fileName):
    '''
    Reads data from a csv and returns it, ready to be plotted.

    Parameters
    ----------
    File Name

    Returns
    ----------
    data - 2D list in the form [all currents in mA, all counts per second]
    '''
    path = 'Data/' + fileName

    file = open(path, 'r')
    csv_file = csv.reader(file)

    rows = []

    for line in csv_file:
        rows.append(line)

    # Converts data back from strings into floats in lists
    data = []
    for row in rows:
        data.append([float(i) for i in row])

    return data


def xy_graph(data):
    '''
    Plots given data as a basic xy graph, without taking the log of the axes.

    Parameters:
    2D list in the form [x,y]

    Returns:
    XY graph
    '''
    plt.scatter(data[0], data[1], marker='.', label='Measured Points')

    # Call the exponential fitting function
    exp_decay_fit(data)

    plt.title('Standard Graph with Exponential Fit')
    plt.xlabel('Current (mA)')
    plt.ylabel('Counts Per Seconds')
    plt.grid(True)

    plt.legend()
    plt.show()


def log_graph(data):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.

    Parameters:
    2D list in the form [x,y]

    Returns:
    Graph with both axes logged to base 10
    '''
    data[0] = np.log10(data[0])
    data[1] = np.log10(data[1])

    plt.plot(data[0], data[1], '.', label='Measured Points')

    linear_fit(data)

    plt.title('Logarithmic Graph')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Seconds')
    plt.grid(True)
    plt.legend()
    plt.show()


def linear_fit(data):
    coef = np.polyfit(data[0], data[1], 1)
    poly1d_fn = np.poly1d(coef)

    plt.plot(data[0], poly1d_fn(data[0]), label='Linear Fit')

def r_squared(data, predicted):
    data[0] = np.array(data[0])

    coef, intercept = np.polyfit(data[0], data[1], 1)
    predicted = coef * data[0] + intercept

    r_squared = r2_score(data[1], predicted)

    return r_squared

def exp_decay_fit(data):
    '''
    Fits the data to the model y = a + b * exp(-x/t), plots the fitted curve, 
    and calculates the R^2 value using sklearn's r2_score.

    Parameters:
    2D list in the form [x, y]
    '''
    # Define the model
    def model(x, a, b, t):
        return a + b * np.exp(-x / t)

    # Normalize data to avoid scaling issues
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
            maxfev=10000  # Increase maximum iterations
        )
        a_fit, b_fit, t_fit = popt
        print(f"Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}, t = {t_fit:.3f}")

        # Rescale parameters to original data
        a_fit_rescaled = a_fit * max(y_data)
        b_fit_rescaled = b_fit * max(y_data)
        t_fit_rescaled = t_fit * max(x_data)

        # Generate predicted y-values for the original x_data
        y_pred_scaled = model(x_data_scaled, a_fit, b_fit, t_fit)
        y_pred = y_pred_scaled * max(y_data)

        # Calculate R^2 value using sklearn's r2_score
        r_squared = r2_score(y_data, y_pred)
        print(f"R^2 value: {r_squared:.3f}")

        # Generate fitted curve for plotting
        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = model(x_fit / max(x_data), a_fit, b_fit, t_fit) * max(y_data)

        # Plot the fitted curve
        plt.plot(x_fit, y_fit, label=f'Fit: a={a_fit_rescaled:.2f}, b={b_fit_rescaled:.2f}, t={t_fit_rescaled:.2f}, R^2={r_squared}', color='green')
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        print("Try adjusting the initial guesses, increasing maxfev, or normalizing your data.")

# Main execution
file = 'B2S61.csv'
data1 = read_data(file)
xy_graph(data1)

