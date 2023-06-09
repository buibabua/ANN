import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.rand(10, 10)

# Create a plot with two subplots
fig, axs = plt.subplots(nrows=1, ncols=2)

# Plot the data on the subplots
im1 = axs[0].imshow(data, cmap='viridis')
im2 = axs[1].imshow(data, cmap='plasma')

# Create two separate color bars
fig.colorbar(im1, ax=axs[0])
fig.colorbar(im2, ax=axs[1])

plt.show()
