import numpy
import MLP



def myfunction(x):
    return x**2 / (numpy.pi * 2)**2

numpy.random.seed(seed=12567)

#
n = 3
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
learning_rate = 0.75
max_iter=12000
network_architecture = [5,3,3]

trained_ann = MLP.MLP.train(x=x,
                    y=y,
                    net_arch=network_architecture,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    activation="sigmoid", # sigmoid or relu
                    GD_type="batch", # stochastic or batch
                    debug=False)

print("\nTraining Time : ", trained_ann["training_time_sec"])
print("Number of Training Iterations : ", trained_ann["elapsed_iter"])
print("Network Architecture : ", trained_ann["net_arch"])
print("Network Error : ", trained_ann["network_error"])

print(">>>> x", x)
predicted_output = MLP.MLP.predict(trained_ann, x)
for i in range(len(y)):
    print("i, x, predicted, init value, theory: ", i, x[i], predicted_output[i], y[i], myfunction(x[i]))

from srxraylib.plot.gol import plot

x1 = []
for i in range(n * 20):
    xi = numpy.random.rand() * numpy.pi * 2
    x1.append([xi])

x1 = numpy.array(x1)
y1 = MLP.MLP.predict(trained_ann, numpy.array(x1))

print(">>>> x", x)

x2 = numpy.linspace(0,2*numpy.pi,100)
plot(x2, myfunction(x2),
    x.flatten(), predicted_output.flatten(),
    x1.flatten(), y1.flatten(),
    legend=["model", "used", "sampled"],
     linestyle=[None,'',''], marker=[None,'x','o'])