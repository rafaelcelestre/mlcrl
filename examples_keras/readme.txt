#
# examples taken from the book:
# Deep Learning With Python
# 2016 Jason Brownlee.
#

# what is not good:
# - use library models and algorithms without explaining what they mean
# - not much said about predictions
# - old code, not following recent changes in libraries (to be corrected by hand)
# - CNNs not so well explained... Code giving errors.

# what is useful:

# - keras-oriented
# - working with examples
# - use scikit-learn to evaluate the model using stratified k-fold cross validation
#   (To use Keras models with scikit-learn, we must use the KerasClassifier wrapper.
#   This class takes a function that creates and returns our neural network model.
#   It also takes arguments that it will pass along to the call to fit()
#   Explains dropout (ch 16)
#   Explain learning rate techniques (ch 17)
#  intro to CNNs