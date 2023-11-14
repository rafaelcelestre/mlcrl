

import numpy
from mlcrl.get_wofry_data import get_wofry_data
from tensorflow.keras import layers
from tensorflow.keras import models

from srxraylib.plot.gol import plot

from keras.models import load_model
import pickle

from keras.optimizers import RMSprop
import json

def get_model(
    architecture = "convnet", # not used!
    kernel_size = (3, 3),
    pool_size = (2, 2),
    activation = 'relu', # 'tanh', #  'softmax'
    padding = 'same',
    input_shape = tuple((256, 64, 1)),
    output_size = 7,
    ):

    model = models.Sequential()

    model.add(layers.Conv2D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding,
                            input_shape=input_shape))
    model.add(layers.Conv2D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool1', pool_size=pool_size))

    model.add(layers.Conv2D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool2', pool_size=pool_size))

    model.add(layers.Conv2D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool3', pool_size=pool_size))

    model.add(layers.Conv2D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool4', pool_size=pool_size))

    model.add(layers.Conv2D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding))
    try:
        if input_shape[0] == 1:
            model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(1, 2)))
        else:
            model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(2, 2)))
    except:
        model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(1, 1)))

    model.add(layers.Flatten(name='flat'))
    model.add(layers.Dense(64, name='dense1', activation=activation))
    model.add(layers.Dense(64, name='dense2', activation=activation))
    model.add(layers.Dense(output_size, name='Y', activation='linear'))

    print(model.summary())
    return model

if __name__ == "__main__":

    dir_wofrydata = "/scisoft/users/srio/ML_TRAIN2_V20/"

    dir_out =       "/scisoft/users/srio/ML_TRAIN2_V20/1000/"  # where the results are going to be written
    only1000 = True

    # dir_out =       "/scisoft/users/srio/ML_TRAIN2_V20/5000/"  # where the results are going to be written
    # only1000 = False
    #
    # dir_out =       "/scisoft/users/srio/ML_TRAIN2_V20/MULTIMODE/"  # where the results are going to be written
    # only1000 = False

    root = "tmp_ml"

    nbin = 1
    (training_data, training_target), (test_data, test_target) = get_wofry_data(root,
                                                                                dir_out=dir_wofrydata,
                                                                                verbose=1,
                                                                                gs_or_z=0,
                                                                                nbin=nbin,     # !!!!!!!!!!!!!! binning  !!!!!!!!!!!
                                                                                only1000=only1000, # !!!!!!!!! cutting N
                                                                                )

    print("Training: ", training_data.shape, training_target.shape)
    print("Test: ", test_data.shape, test_target.shape)

    min_training_data = training_data.min()
    max_training_data = training_data.max()

    print("Min, Max of Training: ", min_training_data, max_training_data)

    # data type: images— 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
    #            could also be Timeseries data or sequence data— 3D tensors of shape (samples, timesteps, features)
    #            right now our data is (samples, features (256), timesteps (65))
    training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], training_data.shape[2], 1))


    training_data = training_data.astype('float32')
    training_data = (training_data - min_training_data) / (max_training_data - min_training_data)

    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
    test_data = test_data.astype('float32')
    test_data = (test_data - min_training_data) / (max_training_data - min_training_data)



    # train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)

    #
    #
    #
    do_train = 1
    model_root = "training_v20"

    if do_train:
        model = get_model(input_shape = tuple((256, 64//nbin, 1)),)

        # To perform regression toward a vector of continuous values, end your stack of layers
        # with a Dense layer with a number of units equal to the number of values you’re trying
        # to predict (often a single one, such as the price of a house), and no activation. Several
        # losses can be used for regression, most commonly mean_squared_error ( MSE ) and
        # mean_absolute_error ( MAE )

        # to choose thecorrect loss:
        # For instance, you’ll use binary crossentropy for a two-class classification
        # problem, categorical crossentropy for a many-class classification problem, mean-
        # squared error for a regression problem, connectionist temporal classification ( CTC )
        # for a sequence-learning problem, and so on.

        model.compile(
                      # optimizer='rmsprop',
                      optimizer=RMSprop(lr=1e-4),
                      loss='mse',
                      # metrics=['mae'], # mean absolute error
                      # loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      )

        # filename = 'training_v1.csv'
        # import tensorflow as tf
        # history_logger = tf.keras.callbacks.CSVLogger(filename, separator=" ", append=False)


        history = model.fit(training_data, training_target,
                            epochs=1500, batch_size=64, validation_split=0.2,
                            # callbacks=[history_logger],
                            )

        model.save('%s/%s.h5' % (dir_out, model_root))


        history_dict = history.history


        with open("%s/%s.json" % (dir_out, model_root), "w") as outfile:
            json.dump(history_dict, outfile)

    else:
        model = load_model('%s/%s.h5' % (dir_out, model_root))

        f = open("%s/%s.json" % (dir_out, model_root), "r")
        f_txt = f.read()
        history_dict = json.loads(f_txt)

    print(history_dict.keys())

    import matplotlib.pyplot as plt
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plot(epochs, loss_values,
         epochs, val_loss_values,
         legend=['loss','val_loss'], xtitle='Epochs', ytitle='Loss', show=0)

    # mae_values = history_dict['mae']
    # val_mae_values = history_dict['val_mae']
    # plot(epochs, mae_values,
    #      epochs, val_mae_values,
    #      legend=['mae','val_mae'], xtitle='Epochs', ytitle='mae')

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plot(epochs, val_acc_values,
         epochs, acc_values,
         legend=['accuracy on validation set','accuracy on training set'],
         color=['g','b'], xtitle='Epochs', ytitle='accuracy')


    # plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # plt.plot(epochs, mae_values, 'b', label='mae')
    # plt.title('Training')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


    #
    # test evaluation
    #
    test_loss, test_acc = model.evaluate(test_data, test_target)
    #
    print(test_acc)


    #
    # predictions
    #
    # predictions = model.predict(test_data)
    # print(predictions.shape)





