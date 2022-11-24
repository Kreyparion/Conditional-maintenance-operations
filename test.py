import matplotlib.pyplot as plt
import numpy as np

def function(x):
    return np.exp(-x/6000)

X = np.linspace(0, 30000, 30000)
Y = function(X)
plt.title("epsilon")
plt.legend("step")
plt.plot(X, Y)
plt.show()