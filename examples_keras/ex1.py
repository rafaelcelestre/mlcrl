
#
# examples taken from the book:
# Deep Learning With Python
# 2016 Jason Brownlee.
#

# chapters < 9


from keras.models import Sequential
from keras.layers import Dense
import numpy



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("diabetes.csv", delimiter=",", skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

print(">>> X: ", X)
print(">>> Y: ", Y)

if True:
    # create model


    # We will use the rectifier (relu) activation function on the first two layers and the sigmoid
    # activation function in the output layer. It used to be the case that sigmoid and tanh activation
    # functions were preferred for all layers. These days, better performance is seen using the rectifier
    # activation function. We use a sigmoid activation function on the output layer to ensure our
    # network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to
    # a hard classification of either class with a default threshold of 0.5. We can piece it all together
    # by adding each layer. The first hidden layer has 12 neurons and expects 8 input variables. The
    # second hidden layer has 8 neurons and finally the output layer has 1 neuron to predict the class
    # (onset of diabetes or not).

    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer="uniform" , activation='relu' ))
    model.add(Dense(8,               kernel_initializer="uniform" , activation='relu' ))
    model.add(Dense(1,               kernel_initializer="uniform" , activation='sigmoid' ))

    # Compile model

    # We must specify the loss function to use to evaluate a set of weights, the optimizer used
    # to search through different weights for the network and any optional metrics we would like
    # to collect and report during training. In this case we will use logarithmic loss, which for a
    # binary classification problem is defined in Keras as binary crossentropy. We will also use the
    # efficient gradient descent algorithm adam for no other reason that it is an efficient default. Learn
    # more about the Adam optimization algorithm in the paper Adam: A Method for Stochastic
    # Optimization 4 . Finally, because it is a classification problem, we will collect and report the
    # classification accuracy as the metric.

    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


    # Fit the model

    # The training process will run for a fixed number of iterations through the dataset called
    # epochs, that we must specify using the nb epoch argument. We can also set the number of
    # instances that are evaluated before a weight update in the network is performed called the
    # batch size and set using the batch size argument. For this problem we will run for a small
    # number of epochs (150) and use a relatively small batch size of 10. Again, these can be chosen
    # experimentally by trial and error.

    do_split = 0 # 0=None, 1=simple, 2=k-fold cross validation

    if do_split ==0:
        history = model.fit(X, Y, epochs=150, batch_size=10)
        # evaluate the model

        # You can evaluate your model on your training dataset using the evaluation() function on
        # your model and pass it the same input and output used to train the model. This will generate a
        # prediction for each input and output pair and collect scores, including the average loss and any
        # metrics you have configured, such as accuracy.

        scores = model.evaluate(X, Y)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    elif do_split == 1:
        # split into 67% for train and 33% for test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
        history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test,y_test))
    elif do_split == 2:
        # In the example below we use the handy StratifiedKFold class 1 from the scikit-learn Python
        # machine learning library to split up the training dataset into 10 folds. The folds are stratified,
        # meaning that the algorithm attempts to balance the number of instances of each class in each
        # fold. The example creates and evaluates 10 models using the 10 splits of the data and collects
        # all of the scores. The verbose output for each epoch is turned off by passing verbose=0 to the
        # fit() and evaluate() functions on the model. The performance is printed for each model and
        # it is stored. The average and standard deviation of the model perfo
        from sklearn.model_selection import StratifiedKFold

        # define 10-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores = []
        for train, test in kfold.split(X, Y):
            # create model
            model2 = Sequential()
            model2.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation= 'relu' ))
            model2.add(Dense(8,               kernel_initializer="uniform", activation= 'relu' ))
            model2.add(Dense(1,               kernel_initializer="uniform", activation= 'sigmoid' ))
            # Compile model
            model2.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
            # Fit the model
            history = model2.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
            # evaluate the model
            scores = model2.evaluate(X[test], Y[test], verbose=0)
            print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    else:
        raise Exception("Why am I here?")


    #
    # save model
    #

    if True:
        from keras.models import model_from_json

        # serialize model to JSON
        model_json = model.to_json(indent=4)
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

#
# load model
#
if False:
    from keras.models import model_from_json

    json_file = open( 'model.json' , 'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss= 'binary_crossentropy' , optimizer= 'rmsprop' , metrics=[ 'accuracy' ])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#
# some plots...
#
if True:
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    # plt.plot(history.history[ 'loss' ])
    plt.plot(history.history[ 'accuracy' ])
    plt.title( 'model accuracy' )
    plt.ylabel( 'accuracy' )
    plt.xlabel( 'epoch' )
    # plt.legend([ 'train' , 'test' ], loc= 'upper left' )
    plt.show()
    # summarize history for loss
    plt.plot(history.history[ 'loss' ])
    # plt.plot(history.history[ 'val_loss' ])
    plt.title( 'model loss' )
    plt.ylabel( 'loss' )
    plt.xlabel( 'epoch' )
    # plt.legend([ 'train' , 'test' ], loc= 'upper left' )
    plt.show()