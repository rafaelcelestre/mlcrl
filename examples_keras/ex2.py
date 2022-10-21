
#
# examples taken from the book:
# Deep Learning With Python
# 2016 Jason Brownlee.
#

# chapters = 9


# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Function to create model, required for KerasClassifier

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer="glorot_uniform" , activation= 'relu' ))
    model.add(Dense(8,               kernel_initializer="glorot_uniform" , activation= 'relu' ))
    model.add(Dense(1,               kernel_initializer="glorot_uniform" , activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

def create_model2(optimizer= 'rmsprop' , kernel_initializer='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=kernel_initializer, activation= 'relu' ))
    model.add(Dense(8,               kernel_initializer=kernel_initializer, activation= 'relu' ))
    model.add(Dense(1,               kernel_initializer=kernel_initializer, activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer=optimizer , metrics=[ 'accuracy' ])
    return model

def section_9_2(): # 9.2 Evaluate Models with Cross Validation
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load pima indians dataset
    dataset = numpy.loadtxt("diabetes.csv", delimiter=",", skiprows=1)
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]


    # create model
    model = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
    # # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

def section_9_3(): # 9.3 Grid Search Deep Learning Model Parameters
    from sklearn.model_selection import GridSearchCV
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load pima indians dataset
    dataset = numpy.loadtxt("diabetes.csv", delimiter=",", skiprows=1)
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]


    # create model
    model = KerasClassifier(model=create_model2, verbose=0)
    # grid search epochs, batch size and optimizer
    optimizers = [ 'rmsprop' , 'adam' ]
    epochs = numpy.array([50, 100, 150])
    batches = numpy.array([5, 10, 20])
    # srio deletest last part (error!)
    kernel_initializer = [ "glorot_uniform" , "normal" , "uniform" ]
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches) #, kernel_initializer=kernel_initializer)


    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    for key in grid.get_params().keys():
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

if __name__ == "__main__":
    # section_9_2()
    section_9_3()