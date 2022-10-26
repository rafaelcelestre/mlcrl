#
# examples taken from the book:
# Deep Learning With Python
# 2016 Jason Brownlee.
#

# chapters = 19


# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

def show_four_images():
    # load (downloaded if needed) the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))
    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap( 'gray' ))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap( 'gray' ))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap( 'gray' ))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap( 'gray' ))
    # show the plot
    plt.show()

# define a simple CNN model
def baseline_model(is_large=0, num_classes=10):

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers.convolutional import Convolution2D
    from keras.layers.convolutional import MaxPooling2D

    if is_large == 0:
        model = Sequential()

        model.add(Convolution2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif is_large ==1:
        model = Sequential()

        model = Sequential()
        model.add(Convolution2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(15, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif is_large == 2: # F Collet's book listing 5.1
        from keras import layers
        from keras import models

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        print(model.summary())
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def standard_NN():
    import numpy
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.utils import np_utils

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    print(">>> train, test: ", len(X_train), len(X_test))
    print(">>>>>y_train: ", y_train)
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]


    # build the model

    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    # original
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Chollet book
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
              verbose=2)
    # Final evaluation of the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(">>> test_acc: %.3f%%" % (test_acc))
    print(">>> Baseline Error: %.2f%%" % (100 - test_acc * 100))



def convolutional_NN(is_large=0):
    # Simple CNN for the MNIST Dataset
    import numpy
    from keras.datasets import mnist
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers import Dropout
    # from keras.layers import Flatten
    # from keras.layers.convolutional import Convolution2D
    # from keras.layers.convolutional import Conv2D
    # from keras.layers.convolutional import MaxPooling2D
    # from keras.utils import np_utils

    from keras.utils import to_categorical

    # from keras import backend as K
    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load data

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("shapes (X_train, y_train), (X_test, y_test): ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # reshape to be [samples][channels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test =  X_test.reshape(X_test.shape[0],   28, 28, 1).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # X_train = X_train.reshape((60000,28,28,1))
    # X_train = X_train.astype('float32') / 255
    # X_test =   X_test.reshape((10000,28,28,1))
    # X_test = X_test.astype('float32') / 255

    print("before categorizing: ", y_train.shape, y_train[0:5])
    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)
    print("after categorizing: ", y_train.shape, y_train[0:5,:])

    # build the model
    model = baseline_model(is_large=is_large, num_classes=y_test.shape[1])

    # Fit the model

    if is_large < 2:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
                  verbose=2)

    else:
        model.fit(X_train, y_train, epochs=5, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))



if __name__ == "__main__":
    # show_four_images()
    # standard_NN()
    # convolutional_NN(is_large=0) # CNN error: 0.91%
    # convolutional_NN(is_large=1) # CNN error: 1.70%
    convolutional_NN(is_large=2) # Chollet  NN error: 0.76%
