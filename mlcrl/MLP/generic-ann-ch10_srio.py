import numpy
import MLP



def myfunction(x):
    return x**2 / (numpy.pi * 2)**2



numpy.random.seed(seed=12567)


# sin
n = 2
x = []
y = []
for i in range(n):
    xi = numpy.random.rand() * numpy.pi * 2
    print(xi)
    yi = myfunction(xi)
    x.append([xi])
    y.append([yi])

# print(y)
x = numpy.array(x)
y = numpy.array(y)
# print(x,y)
learning_rate = 0.5
max_iter=12000 * 2
network_architecture = [8,5]

# this one does not work...
# x = numpy.array([[0, 0],
#                  [0, 1],
#                  [1, 0],
#                  [1, 1]])
#
# y = numpy.array([[0.],
#                  [1.],
#                  [1.],
#                  [0.]])
# network_architecture = [7, 5, 4]
# learning_rate=0.7
# max_iter=500


# #chapter 8
# x = numpy.array([[0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6]])
# y = numpy.array([[0.45]])
# learning_rate = 0.001
# max_iter=12000
# network_architecture = [8, 5, 3]


#
# network_architecture = [5]
#
# trained_ann = MLP.MLP.train(x=x,
#                             y=y,
#                             net_arch=network_architecture,
#                             max_iter=5000,
#                             learning_rate=0.7,
#                             # activation="sigmoid",
#                             # GD_type="batch",
#                             debug=False)

# x = numpy.array([[0.1, 0.4, 4.1]])
# y = numpy.array([[0.2]])

# network_architecture defines the number of hidden neurons in the hidden layers. It must be a list not any other datatype.
# network_architecture = [7, 5, 4]

# Network Parameters
trained_ann = MLP.MLP.train(x=x,
                    y=y,
                    net_arch=network_architecture,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    # activation="sigmoid",
                    # GD_type="batch",
                    debug=False)

print("\nTraining Time : ", trained_ann["training_time_sec"])
print("Number of Training Iterations : ", trained_ann["elapsed_iter"])
print("Network Architecture : ", trained_ann["net_arch"])
print("Network Error : ", trained_ann["network_error"])

print(">>>> x", x)
predicted_output = MLP.MLP.predict(trained_ann, x)
for i in range(len(y)):
    print("i, x, predicted, init value, theory: ", i, x[i], predicted_output[i], y[i], myfunction(x[i]))
# print("\nPredicted Output(s) : ", MLP.MLP.predict(trained_ann, x))
# print("\nExact Output(s) : ", y)

from srxraylib.plot.gol import plot

x1 = []
# y1 = []
for i in range(n):
    xi = numpy.random.rand() * numpy.pi * 2
    # print(xi)
    # yi = MLP.MLP.predict(trained_ann, numpy.array([x1]))
    x1.append([xi])
    # y1.append([yi])

x1 = numpy.array(x1)
print(numpy.array(x1).shape, numpy.array(x1))
# y1 = numpy.array(y1)
y1 = MLP.MLP.predict(trained_ann, numpy.array(x1))

print(">>>> x", x)

print(">>>>><<", x[0], predicted_output[0])
print(">>>>><<x", x.flatten(), predicted_output.flatten())
print(">>>>><<x1", x1.flatten(), y1.flatten())

x2 = numpy.linspace(0,2*numpy.pi,100)
plot(x2, myfunction(x2),
    x.flatten(), predicted_output.flatten(),
    x1.flatten(), y1.flatten(),
    legend=["model", "used", "sampled"],
     linestyle=[None,'',''], marker=[None,'x','o'])