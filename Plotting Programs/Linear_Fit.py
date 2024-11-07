import pandas 
import csv
import matplotlib.pyplot as plt

fileName = 'BIGDATA.csv'

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

    plt.plot(data[0],data[1])

    plt.title('Standard Graph')
    plt.xlabel('Current (mA)')
    plt.ylabel('Counts Per Seconds')
    plt.grid(True)

    plt.show()




def log_graph(data):
    '''
    Plots given data as a basic xy graph, while taking the log of the axes.

    Paramters:
    2D list in the form [x,y]

    Returns
    Graph with both axes logged to base 10
    '''

    plt.plot(data[0],data[1])


    plt.title('Logarithmic Graph')
    plt.xlabel('Log of Current (mA)')
    plt.ylabel('Log of Counts Per Seconds')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    
    plt.show()
   



data = read_data(fileName)
log_graph(data)
xy_graph(data)