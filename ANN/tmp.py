import numpy as np
from scipy.optimize import fmin

my_array = np.array([[[ 0.46367723,  0.81660696,  0.70801   ,  0.37138836,  2.13621263,
         1.33229428,  0.54713574,  0.08002742,  1.11374133,  0.32885784],
       [-0.2068852 , -0.63747612,  1.40231553,  1.40534946,  0.0838423 ,
         0.67841434, -2.13929369,  0.36391537,  1.20281224,  0.45715   ]],
      [[-0.53027066, -1.25676831],
       [ 0.02761009,  0.09378584],
       [ 0.72847802, -1.03660028],
       [-1.33389713,  1.09000686],
       [-0.63975987,  0.82320563],
       [-0.41630338, -0.15156257],
       [ 0.33922558, -0.22326354],
       [-1.5460348 , -0.96022606],
       [-0.04502939, -0.57847551],
       [-0.38515078,  1.04991733]],
      [[-1.24768029,  1.26366356],
       [-1.54473117, -1.49782204]],
      [np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
       np.array([[0., 0.]]),
       np.array([[0., 0.]])]])

# Flatten the array
initial_guess = my_array.flatten()

# Define the function to be minimized
def my_func(x):
    return np.sum(x**2)

# Call scipy.optimize.fmin with the function and the initial guess
minimum = fmin(my_func, initial_guess)

# Reshape the minimum value to match the original array shape
minimum_array = minimum.reshape(my_array.shape)