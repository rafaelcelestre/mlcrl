
# https://github.com/fchollet/deep-learning-with-python-notebooks


In classical programming, the paradigm of symbolic AI, humans input rules (a program) and data to
be processed according to these rules, and out come answers.

With machine learning, humans input data as well as the answers expected from the data, and out come the rules.


Machine learning is tightly related
to mathematical statistics, but it differs from statistics in several important ways.
Unlike statistics, machine learning tends to deal with large, complex datasets (such as
a dataset of millions of images, each consisting of tens of thousands of pixels) for
which classical statistical analysis such as Bayesian analysis would be impractical. As a
result, machine learning, and especially deep learning, exhibits comparatively little
mathematical theory—maybe too little—and is engineering oriented. It’s a hands-on
discipline in which ideas are proven empirically more often than theoretically.


So that’s what deep learning is, technically: a multistage way to learn data representa-
tions. It’s a simple idea—but, as it turns out, very simple mechanisms, sufficiently
scaled, can end up looking like magic.

If deep learning is your
first contact with machine learning, then you may find yourself in a situation where all
you have is the deep-learning hammer, and every machine-learning problem starts to
look like a nail. The only way not to fall into this trap is to be familiar with other
approaches and practice them when appropriate.

Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theo-
rem while assuming that the features in the input data are all independent (a strong,
or “naive” assumption, which is where the name comes from). This form of data analy-
sis predates computers and was applied by hand decades before its first computer
implementation (most likely dating back to the 1950s).


two approaches: gradient boosting
machines and deep learning. Specifically, gradient boosting is used for problems
where structured data is available, whereas deep learning is used for perceptual prob-
lems such as image classification. Practitioners of the former almost always use the
excellent XGBoost library, which offers support for the two most popular languages of
data science: Python and R. Meanwhile, most of the Kaggle entrants using deep learn-
ing use the Keras library, due to its ease of use, flexibility, and support of Python.
These are the two techniques you should be the most familiar with in order to be
successful in applied machine learning today: gradient boosting machines, for shallow-
learning problems; and deep learning, for perceptual problems. In technical terms,
this means you’ll need to be familiar with XGB oost and Keras—the two libraries that
currently dominate Kaggle competitions.

Machine learning isn’t mathematics or physics, where major advances can be done with
a pen and a piece of paper. It’s an engineering science.


ImageNet dataset: consisting of 1.4 million images that have been hand annotated
with 1,000 image categories (1 category per image).
As Kaggle has been demonstrating since 2010, public competitions are an excel-
lent way to motivate researchers and engineers to push the envelope. Having common
benchmarks that researchers compete to beat has greatly helped the recent rise of
deep learning.

[NN] weren’t able to shine against
more-refined shallow methods such as SVM s and random forests. The key issue was that
of gradient propagation through deep stacks of layers. The feedback signal used to train
neural networks would fade away as the number of layers increased.

This changed around 2009–2010 with the advent of several simple but important
algorithmic improvements that allowed for better gradient propagation:
- Better activation functions for neural layers
- Better weight-initialization schemes, starting with layer-wise pretraining, which was
quickly abandoned
- Better optimization schemes, such as RMSP rop and Adam

Only when these improvements began to allow for training models with 10 or more
layers did deep learning start to shine.
Finally, in 2014, 2015, and 2016, even more advanced ways to help gradient propa-
gation were discovered, such as batch normalization, residual connections, and depth-
wise separable convolutions. Today we can train from scratch models that are
thousands of layers deep.

Nowadays, basic Python scripting skills suffice to do advanced deep-learning research. This has been
driven most notably by the development of Theano and then TensorFlow—two symbolic
tensor-manipulation frameworks for Python that support autodifferentiation, greatly sim-
plifying the implementation of new models—and by the rise of user-friendly libraries
such as Keras, which makes deep learning as easy as manipulating LEGO bricks.

[For handwriting example,] the test-set accuracy turns out to be 97.8%—that’s quite a bit lower than the training
set accuracy. This gap between training accuracy and test accuracy is an example of
overfitting: the fact that machine-learning models tend to perform worse on new data
than on their training data.

Nowadays, and for years to come, people will implement networks in modern
frameworks that are capable of symbolic differentiation, such as TensorFlow. This means
that, given a chain of operations with a known derivative, they can compute a gradient
function for the chain (by applying the chain rule) that maps network parameter values
to gradient values. When you have access to such a function, the backward pass is
reduced to a call to this gradient function. Thanks to symbolic differentiation, you’ll
never have to implement the Backpropagation algorithm by hand. For this reason, we
won’t waste your time and your focus on deriving the exact formulation of the Back-
propagation algorithm in these pages. All you need is a good understanding of how
gradient-based optimization works.

Ch 3
====

A deep-learning model is a directed, acyclic graph of layers. The most common
instance is a linear stack of layers, mapping a single input to a single output.

But as you move forward, you’ll be exposed to a much broader variety of network
topologies. Some common ones include the following:
- Two-branch networks
- Multihead networks
- Inception blocks
The topology of a network defines a hypothesis space.

Picking the right network architecture is more an art than a science; and although
there are some best practices and principles you can rely on, only practice can help
you become a proper neural-network architect.

Once the network architecture is defined, you still have to choose two more things:
- Loss function (objective function)—The quantity that will be minimized during
training. It represents a measure of success for the task at hand.
- Optimizer—Determines how the network will be updated based on the loss func-
tion. It implements a specific variant of stochastic gradient descent ( SGD ).

Fortunately, when it comes to common problems such as classification, regression,
and sequence prediction, there are simple guidelines you can follow to choose the
correct loss. For instance, you’ll use binary crossentropy for a two-class classification
problem, categorical crossentropy for a many-class classification problem, mean-
squared error for a regression problem, connectionist temporal classification ( CTC )
for a sequence-learning problem, and so on. Only when you’re working on truly new
research problems will you have to develop your own objective functions.

A relu (rectified linear unit) is a function meant to zero out negative values
(see figure 3.4), whereas a sigmoid “squashes” arbitrary values into the [0, 1] interval
(see figure 3.5), outputting something that can be interpreted as a probability.

Without an activation function like relu (also called a non-linearity), the Dense layer
would consist of two linear operations so the layer could only learn linear transformations
(affine transformations) of the input data. In order to get access to a much richer hypothesis
space that would benefit from deep representations, you need a non-linearity, or activation function.
relu (rectified linear unit function) is the most popular activation function in deep learning,
but there are many other candidates, which all come with similarly strange names: prelu, elu, and so on.

optimizer='rmsprop is equivalent to optimizer=optimizers.RMSprop(lr=0.001)
loss='binary_crossentropy' is equivalent to loss=losses.binary_crossentropy
metrics=['accuracy'] is equivalent to metrics=[metrics.binary_accuracy]


As you can see [Figs. 3.7,3.8], the training loss decreases with every epoch, and the training accuracy
increases with every epoch. That’s what you would expect when running gradient-
descent optimization—the quantity you’re trying to minimize should be less with
every iteration. But that isn’t the case for the validation loss and accuracy: they seem to
peak at the fourth epoch. This is an example of what we warned against earlier: a
model that performs better on the training data isn’t necessarily a model that will do
better on data it has never seen before. In precise terms, what you’re seeing is overfit-
ting : after the second epoch, you’re overoptimizing on the training data, and you end
up learning representations that are specific to the training data and don’t generalize
to data outside of the training set.

3.5 Multiclass classification

To vectorize the labels, there are two possibilities: you can cast the label list as an inte-
ger tensor, or you can use one-hot encoding. One-hot encoding is a widely used for-
mat for categorical data, also called categorical encoding. For a more detailed
explanation of one-hot encoding, see section 6.1. In this case, one-hot encoding of
the labels consists of embedding each label as an all-zero vector with a 1 in the place of
the label index. ... Another way to encode the labels would be to cast them as
an integer tensor.
The only thing this approach would change is the choice of the loss function. The loss
function used in listing 3.21, categorical_crossentropy , expects the labels to follow
a categorical encoding. With integer labels, you should use sparse_categorical_
crossentropy.

Here's what you should take away from this example:

- If you are trying to classify data points between N classes, your network should end with a Dense layer of size N.
- In a single-label, multi-class classification problem, your network should end with a softmax activation, so that it will output a probability distribution over the N output classes.
- Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the network, and the true distribution of the targets.
- There are two ways to handle labels in multi-class classification: ** Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using categorical_crossentropy as your loss function. ** Encoding the labels as integers and using the sparse_categorical_crossentropy loss function.
- If you need to classify data into a large number of categories, then you should avoid creating information bottlenecks in your network by having intermediate layers that are too small.

3.6 Regression

Here’s what you should take away from this example:
- Regression is done using different loss functions than what we used for classifi-
   cation. Mean squared error ( MSE ) is a loss function commonly used for regression.
- Similarly, evaluation metrics to be used for regression differ from those used for
  classification; naturally, the concept of accuracy doesn’t apply for regression. A
  common regression metric is mean absolute error ( MAE ).
- When features in the input data have values in different ranges, each feature
  should be scaled independently as a preprocessing step.
- When there is little data available, using K-fold validation is a great way to reli-
  ably evaluate a model.
- When little training data is available, it’s preferable to use a small network with
  few hidden layers (typically only one or two), in order to avoid severe overfitting.

Ch 4 Fundamentals of ML
=======================

Supervised learning
  - binary classification
  - multiclass classification
  - multilabek classification
  - scalar regression
  - Vector regression: A task where the target is a set of continuous values
  - Others:
    - Sequence generation
    - Syntax tree prediction
    - Object detection
    - Image segmentation

Unsupervised learning
  - Dimensionality reduction
  - clustering

Self-supervised learning
  This is a specific instance of supervised learning, but it’s different enough that it
  deserves its own category. There are still labels involved (because the learning has to be
  supervised by something), but they’re generated from the input data, typically using a
  heuristic algorithm.
  Self-supervised learning is supervised learning without
  human-annotated labels—you can think of it as supervised learning without any
  humans in the loop.

Reinforcement learning



overfit: the [model] performance on never-before-seen data started stalling
(or worsening) compared to their performance on the training data—which always
improves as training progresse

In machine learning, the goal is to achieve models that generalize—that perform
well on never-before-seen data—and overfitting is the central obstacle. You can only
control that which you can observe, so it’s crucial to be able to reliably measure the
"generalization" power of your model.

Splitting your data into training, validation, and test sets may seem straightforward,
but there are a few advanced ways to do it that can come in handy when little data is
available.


ITERATED K- FOLD VALIDATION WITH SHUFFLING:
It consists of applying K -fold validation multiple times, shuffling
the data every time before splitting it K ways. The final score is the average of the
scores obtained at each run of K -fold validation. Note that you end up training and
evaluating P × K models (where P is the number of iterations you use), which can very
expensive.

The processing of fighting overfitting this way is called regularization.
The simplest way to prevent overfitting is to reduce the size of the model: the number
of learnable parameters in the model (which is determined by the number of layers
and the number of units per layer).
Always keep this in mind: deep-
learning models tend to be good at fitting to the training data, but the real challenge
is generalization, not fitting.
In deep learning, the number of learnable parameters in a model is often referred to
as the model’s "capacity".
There is a compromise to be found between too much capacity and not enough capacity.

Unfortunately, there is no magical formula to determine the right number of lay-
ers or the right size for each layer. You must evaluate an array of different architec-
tures (on your validation set, not on your test set, of course) in order to find the
correct model size for your data.

Table 4.1
Choosing the right last-layer activation and loss function for your model

Problem type                              Last-layer activation    Loss function
------------                              ---------------------    -------------
Binary classification                     sigmoid                  binary_crossentropy
Multiclass, single-label classification   softmax                  categorical_crossentropy
Multiclass, multilabel classification     sigmoid                  binary_crossentropy
Regression to arbitrary values            None                     mse
Regression to values between 0 and 1      sigmoid                  mse or binary_crossentropy