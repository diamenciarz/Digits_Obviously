import numpy as np
import matplotlib.pyplot as plt

def correct_type(data):
    corrected = np.zeros([data.shape[0], data.shape[1]])
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            corrected[y, x] = float(data[y, x])
    return corrected

def displayData(data):
    fig = plt.figure()
    plt.gray()
    for i in range(data.shape[0]):
        subplot_dimensions = int(np.ceil(np.sqrt(data.shape[0])))
        ax = fig.add_subplot(subplot_dimensions, subplot_dimensions, i+1)
        reshaped = correct_type(np.reshape(data[i], (-1, 20)))
        ax.imshow(reshaped.T)
        ax.set_axis_off()
    plt.show()
