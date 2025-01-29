import csv
import matplotlib.pyplot as plt

fileName = 't_metric.csv'
path = 'Data/' + fileName

with open(path, 'r') as file:
    csv_file = csv.reader(file)
    rows = [line for line in csv_file]

concentration = []
t_metric = []

for row in rows:
    try:
        concentration.append(float(row[1]))
        t_metric.append(float(row[0]))
    except:
        pass

plt.scatter(concentration,t_metric,label = 't')
plt.xlabel('Concentration (%)')
plt.ylabel('t')
plt.show()
