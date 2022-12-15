

import numpy
from mlcrl.get_wofry_data import get_wofry_data
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from srxraylib.plot.gol import plot

import keras
from keras.models import load_model
import pickle

from keras.optimizers import RMSprop
import json
import h5py


# https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, filenames, targets, batch_size):
        self.filenames = filenames
        self.targets = targets
        self.batch_size = batch_size

    def __len__(self):
        return (numpy.ceil(len(self.filenames) / float(self.batch_size))).astype(numpy.int)

    def __getitem__(self, idx):
        batch_x = numpy.array(self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size])
        batch_y = numpy.array(self.targets[idx * self.batch_size: (idx + 1) * self.batch_size])
        # print(">>> batch_x.shape", batch_x.shape)
        # print(">>> batch_y.shape", batch_y.shape)



        stack =  numpy.zeros((self.batch_size, 64, 256, 256))
        for i,filename in enumerate(batch_x):
            # print(">>>>> reading file", i, filename[0])
            f = h5py.File(filename[0],'r')
            Z = f['singlesample/intensity/stack_data'][()]
            f.close()
            stack[i] = Z * 1e-20

        # min_training_data = training_data.min()
        # max_training_data = training_data.max()
        #
        # print("Min, Max of Training: ", min_training_data, max_training_data)
        #
        # # data type: images— 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
        # #            could also be Timeseries data or sequence data— 3D tensors of shape (samples, timesteps, features)
        # #            right now our data is (samples, features (256), timesteps (65))
        # training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], training_data.shape[2], 1))
        #
        #
        # training_data = training_data.astype('float32')
        # training_data = (training_data - min_training_data) / (max_training_data - min_training_data)

        return stack, batch_y



def get_model(
    architecture = "convnet", # not used!
    kernel_size = (3, 3, 3),
    pool_size = (1, 2, 2),
    activation = 'relu', # 'tanh', #  'softmax'
    padding = 'same',
    input_shape = tuple((64, 256, 256)),
    output_size = 7,
    ):

    model = models.Sequential()

    model.add(layers.Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding,
                            input_shape=input_shape))
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


    # try:
    #     if input_shape[0] == 1:
    #         model.add(layers.MaxPooling3D(name='maxpool5', pool_size=(1, 2, 2)))
    #     else:
    #         model.add(layers.MaxPooling3D(name='maxpool5', pool_size=(2, 2, 2)))
    # except:
    #     model.add(layers.MaxPooling3D(name='maxpool5', pool_size=(1, 1, 1)))
    model.add(layers.MaxPooling3D(name='maxpool5', pool_size=pool_size))

    model.add(layers.Conv3D(256, name='conv11', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv3D(256, name='conv12', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling3D(name='maxpool6', pool_size=pool_size))

    model.add(layers.Conv3D(512, name='conv13', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv3D(512, name='conv14', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling3D(name='maxpool7', pool_size=pool_size))


    model.add(layers.Conv3D(1024, name='conv15', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv3D(1024, name='conv16', kernel_size=kernel_size, activation=activation, padding=padding))

    try:
        if input_shape[0] == 1:
            model.add(layers.MaxPooling3D(name='maxpool8', pool_size=(1, 2, 2)))
        else:
            model.add(layers.MaxPooling3D(name='maxpool8', pool_size=(2, 2, 2)))
    except:
        model.add(layers.MaxPooling3D(name='maxpool8', pool_size=(1, 1, 1)))

    model.add(layers.Flatten(name='flat'))
    model.add(layers.Dense(64, name='dense1', activation=activation))
    model.add(layers.Dense(64, name='dense2', activation=activation))
    model.add(layers.Dense(output_size, name='Y', activation='linear'))

    print(model.summary())
    return model

if __name__ == "__main__":
    dir_data = "/scisoft/users/srio/ML_TRAIN2/RESULTS_2D/"
    dir_out = "/scisoft/users/srio/ML_TRAIN2/RESULTS_2D_FIT/"  # where the results are going to be written
    root = "tmp_ml"

    targets = numpy.loadtxt("%s/%s_targets_z.txt" % (dir_data, root))[:,1:].copy()
    print(targets.shape)
    print(targets[0:2,:])

    filenames = []
    for i in range(5000):
        filenames.append("%s/%s_%06d.h5" % (dir_data, root, i))

    filenames = numpy.array(filenames)
    filenames.shape = (filenames.size,1)
    print(filenames.shape, filenames[10])

    # numpy.save('targets.npy', targets)
    # numpy.save('filenames.npy', filenames)



    print("filenames.shape" , filenames.shape)
    print("targets.shape" , targets.shape)

    if False:
        from sklearn.model_selection import train_test_split
        filenames_train, filenames_val, targets_train, targets_val = train_test_split(
            filenames, targets, test_size=0.2, shuffle=False, random_state=None)
    else:
        filenames_train = filenames[0:4000,:].copy()
        filenames_val = filenames[4000:,:].copy()
        targets_train = targets[0:4000,:].copy()
        targets_val = targets[4000:,:].copy()



    if True:
        print("\n** filenames_train.shape" , filenames_train.shape, filenames_train[0:2])
        print("\n** filenames_val.shape" , filenames_val.shape, filenames_val[0:2])
        print("\n** targets_train.shape" , targets_train.shape, targets_train[0:2])
        print("\n** targets_val.shape" , targets_val.shape, targets_val[0:2])

    batch_size = 4
    my_training_batch_generator = My_Custom_Generator(filenames_train, targets_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(filenames_val, targets_val, batch_size)

        # train0 = my_training_batch_generator[0]
        # print(  train0[0].shape, train0[1].shape, train0[1][0,:])


    #
    #
    #
    do_train = 1
    model_root = "training_v30"

    if do_train:
        model = get_model(input_shape = tuple((64, 256, 256)),)

        model.compile(
                      # optimizer='rmsprop',
                      optimizer=RMSprop(lr=1e-4),
                      loss='mse',
                      # metrics=['mae'], # mean absolute error
                      # loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      )
    #
        # filename = 'training_v1.csv'
    #     # import tensorflow as tf
    #     history_logger = tf.keras.callbacks.CSVLogger(filename, separator=" ", append=False)

    #
    #     history = model.fit(training_data, training_target,
    #                         epochs=1500, batch_size=64, validation_split=0.2,
    #                         # callbacks=[history_logger],
    #                         )
    #

        history = model.fit_generator(generator=my_training_batch_generator,
                            steps_per_epoch=int(0.8 * 5000 // batch_size),
                            epochs=10,
                            verbose=1,
                            validation_data=my_validation_batch_generator,
                            validation_steps=int(0.2 * 5000 // batch_size),
                            # callbacks=[history_logger],
                                      )



        model.save('%s.h5' % model_root)
    #
    #
        history_dict = history.history
    #
    #
    #     with open("%s.json" % model_root, "w") as outfile:
    #         json.dump(history_dict, outfile)
    #
    # else:
    #     model = load_model('%s/%s.h5' % (dir_out, model_root))
    #
    #     f = open("%s/%s.json" % (dir_out, model_root), "r")
    #     f_txt = f.read()
    #     history_dict = json.loads(f_txt)
    #
    print(history_dict.keys())
    #
    import matplotlib.pyplot as plt
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plot(epochs, loss_values,
         epochs, val_loss_values,
         legend=['loss','val_loss'], xtitle='Epochs', ytitle='Loss', show=0)
    #
    # # mae_values = history_dict['mae']
    # # val_mae_values = history_dict['val_mae']
    # # plot(epochs, mae_values,
    # #      epochs, val_mae_values,
    # #      legend=['mae','val_mae'], xtitle='Epochs', ytitle='mae')
    #
    # acc_values = history_dict['accuracy']
    # val_acc_values = history_dict['val_accuracy']
    # plot(epochs, val_acc_values,
    #      epochs, acc_values,
    #      legend=['accuracy on validation set','accuracy on training set'],
    #      color=['g','b'], xtitle='Epochs', ytitle='accuracy')
    #
    #
    # # plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # # plt.plot(epochs, mae_values, 'b', label='mae')
    # # plt.title('Training')
    # # plt.xlabel('Epochs')
    # # plt.ylabel('Loss')
    # # plt.legend()
    # # plt.show()
    #
    #
    # #
    # # test evaluation
    # #
    # test_loss, test_acc = model.evaluate(test_data, test_target)
    # #
    # print(test_acc)
    #
    #
    # #
    # # predictions
    # #
    # # predictions = model.predict(test_data)
    # # print(predictions.shape)
    #
    #
    #
    #
    #
