
from tensorflow.keras import layers
from tensorflow.keras import models

architecture = "convnet"
kernel_size = (3, 3, 3)
pool_size = (1, 2, 2)
activation = 'tanh'
padding = 'same'

input_shape = tuple((32,32,32, 1))
output_size = 11

model = models.Sequential()

# def _convnet(self, input_shape, output_size, kernel_size, activation, padding, pool_size):
# inp = layers.Input(name='X', shape=input_shape)
model.add(layers.Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding, input_shape=input_shape))
model.add(layers.Conv3D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.MaxPooling3D(name='maxpool1', pool_size=pool_size))

model.add(layers.Conv3D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.Conv3D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.MaxPooling3D(name='maxpool2', pool_size=pool_size))

model.add(layers.Conv3D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.Conv3D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.MaxPooling3D(name='maxpool3', pool_size=pool_size))

model.add(layers.Conv3D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.Conv3D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.MaxPooling3D(name='maxpool4', pool_size=pool_size))

model.add(layers.Conv3D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding))
model.add(layers.Conv3D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding))
if input_shape[0] == 1:
    model.add(layers.MaxPooling3D(name='maxpool5', pool_size=(1, 2, 2)))
else:
    model.add(layers.MaxPooling3D(name='maxpool5', pool_size=(2, 2, 2)))

model.add(layers.Flatten(name='flat'))
model.add(layers.Dense(64, name='dense1', activation=activation))
model.add(layers.Dense(64, name='dense2', activation=activation))
model.add(layers.Dense(output_size, name='Y', activation='linear'))

# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding))
# model.add(layers.Conv3D(8, name='conv1', kernel_size=kernel_size, input_shape=input_shape))
# model.build(input_shape)
print(model.summary())
# model = models.Model(inputs=inp, outputs=oup)