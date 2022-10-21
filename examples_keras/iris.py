#
# examples taken from the book:
# Deep Learning With Python
# 2016 Jason Brownlee.
#

# chapter = 10

# Multiclass Classification with the Iris Flowers Dataset
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer= 'normal' , activation= 'relu' ))
    model.add(Dense(3,              kernel_initializer= 'normal' , activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model



from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("Iris.csv", header=None, skiprows=1)
dataset = dataframe.values

Y = dataset[:,5]
X = dataset[:,1:5].astype(float)

print(">>>>Y: ", Y)
print(">>>>X: ", X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(">>>> encoded Y: ", encoded_Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(">>>> dummy_y: ", dummy_y)



# define baseline model
estimator = KerasClassifier(model=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))