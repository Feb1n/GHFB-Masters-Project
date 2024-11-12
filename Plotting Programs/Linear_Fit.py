import pandas 
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def read_data(fileName):
    '''
    Reads data from a csv and returns it, ready to be plotted.

    Parameters
    ----------
    File Name

    Returns
    ----------
    data - 2d list in the form [all currents in mA, all counts per second]
    
    '''
    path = 'Data/'+fileName

    file = open(path,'r') 
    csv_file = csv.reader(file)

    rows = []

    for line in csv_file:
        rows.append(line)

    # converts data back from strings into integers in lists

    data =[]
    for row in rows:
        data.append([float(i) for i in row])

    return data


def xy_graph(data):
    '''
    Plots given data as a basic xy graph, without taking the log of the axes.

    Paramters:
    2D list in the form [x,y]

    Returns
    XY graph
    '''

    plt.plot(data[0],data[1],marker = '.',label = 'Measured Points')
    
    r_2 = r_squred(data)

    plt.title('Standard Graph')
    plt.xlabel('Current (mA)')
    plt.ylabel('Counts Per Seconds')
    plt.grid(True)

    plt.legend()
    plt.show()
    plt.annotate('r^2 = {}'.format(r_2),xy = (min(data[0]),0.9*max(data[1])))



def log_graph(data):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.

    Paramters:
    2D list in the form [x,y]

    Returns
    Graph with both axes logged to base 10
    '''

    data[0] = np.log10(data[0])
    data[1] = np.log10(data[1])

    plt.plot(data[0],data[1], '.',label = 'Measured Points')

    linear_fit(data)
    r_2 = r_squred(data)
    print(r_2)
    
    plt.title('Logarithmic Graph')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Seconds')
    plt.grid(True)
    plt.annotate('r^2 = {}'.format(r_2),xy = (min(data[0]),0.9*max(data[1])))
    plt.legend()
    plt.show()
   
def double_data_plot(data1,data2):
    plt.plot(data1[0],data1[1],label = 'Ratio of sample and reference')
    plt.plot(data2[0],data2[1],label = 'Ratio of references')
    plt.title('Standard Graph')
    plt.xlabel('Current (mA)')
    plt.ylabel('Ratio of Counts Per Seconds')
    plt.grid(True)
    plt.legend()

    plt.show()

def r_squred(data):
    data[0] = np.array(data[0])
    coef,intercept = np.polyfit(data[0],data[1],1)
    predicted = coef*data[0]+intercept
    r_squared = r2_score(data[1], predicted)

    return r_squared

def linear_fit(data):
    coef = np.polyfit(data[0],data[1],1)
    poly1d_fn = np.poly1d(coef) 

    plt.plot(data[0], poly1d_fn(data[0]),label = 'Linear fit')



file = 'B1S11.csv'
data1 = read_data(file)
xy_graph(data1)
