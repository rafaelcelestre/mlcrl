# see https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/
import numpy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

print ('TensorFlow version: ' + tf.__version__)

def myfunc(x, noise=0.0):
    y = 0.1 * x *np.cos(x)
    # return np.cos(x)
    if noise > 0.0:
        y += noise * np.random.normal(size=x.size)

    return y





# Create noisy data
x_data = np.linspace(-10, 10, num=1000)
y_data = myfunc(x_data, noise=0.1)
print('Data created successfully')


# Display the dataset
plt.scatter(x_data[::1], y_data[::1], s=2)
plt.grid()
plt.show()



# # Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

# Display the model
model.summary()

# Training
model.fit( x_data, y_data, epochs=100, verbose=1)

# Compute the output
# y_predicted = model.predict(x_data)
# Display the result
# plt.scatter(x_data[::1], y_data[::1], s=1)
# plt.plot(x_data, model.predict(x_data), 'r', linewidth=4)
# plt.grid()
# plt.show()

from srxraylib.plot.gol import plot
print()
xx = numpy.linspace(-10,10,300)
plot(x_data, y_data,
     x_data, model.predict(x_data),
     xx, myfunc(xx, noise=0),
     xx, model.predict(xx),
     legend=["scatter data", "predicted on model x", "exact on model", "full predictions"],
     linestyle=['', None, None, None],
     marker=['+', None, None, None])


# plt.savefig('training.png', dpi=300)
# files.download("training.png")

# create image sequence for video
# for x in range(100):
#   # One epoch
#   model.fit( x_data, y_data, epochs=1, verbose=1)
#
#   # Compute the output
#   y_predicted = model.predict(x_data)
#
#   # Display the result
#   plt.scatter(x_data[::1], y_data[::1], s=2)
#   plt.plot(x_data, y_predicted, 'r', linewidth=4)
#   plt.grid()
#   plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
#   plt.ylim(bottom=-1.2)
#   #plt.show()
#   plt.savefig('training-' + str(x) +'-epochs.png',dpi=300)
#   files.download('training-' + str(x) +'-epochs.png')
#   plt.clf()



