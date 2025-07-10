import matplotlib.pyplot as plt 
import numpy as np 
import os 

script_path = os.path.realpath(__file__)

# Extract the directory name from the script's path
script_directory = os.path.dirname(script_path)

filename = f"{script_directory}/../build/data.csv"
data = np.genfromtxt(filename, delimiter=",")

for d in data:
    plt.plot(*d[0:2], 'r.')
    plt.plot(*d[2:4], 'b.')
    plt.plot(*(d[0:2] + d[4:6]), 'g.')
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])
    plt.show()