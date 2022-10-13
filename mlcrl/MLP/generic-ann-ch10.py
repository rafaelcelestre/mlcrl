import numpy
import MLP
######################################################################
# # it does not work...
# x = numpy.array([[0, 0],
#                  [0, 1],
#                  [1, 0],
#                  [1, 1]])
#
# y = numpy.array([[0],
#                  [1],
#                  [1],
#                  [0]])
#
# network_architecture = [2]
#
# trained_ann = MLP.MLP.train(x=x,
#                             y=y,
#                             net_arch=network_architecture,
#                             max_iter=5000,
#                             learning_rate=1,
#                             activation="sigmoid",
#                             GD_type="batch",
#                             debug=True)

######################################################################
# x = numpy.array([[.1,.4,4.1]])
#
# y = numpy.array([[0.05,0.2]])
#
# network_architecture = [5]
#
# trained_ann = MLP.MLP.train(x=x,
#                             y=y,
#                             max_iter=80000,
#                             net_arch=network_architecture,)

######################################################################
# x = numpy.array([[.1,.4,4.1]])
#
# y = numpy.array([[0.05,0.2,0.9]])
#
# network_architecture = [5]
#
# trained_ann = MLP.MLP.train(x=x,
#                             y=y,
#                             max_iter=20000,
#                             learning_rate=0.9,
#                             net_arch=network_architecture,
#                             activation="sigmoid", # "sigmoid", ("relu" does not work)
#                             debug=False)


######################################################################
x = numpy.array([[0.1, 0.5, 0.1],
                 [0.4, 0.8, 0.4],
                 [0.7, 0.7, 0.7] ])

y = numpy.array([[0.9],
                 [0.5],
                 [0.1]])

network_architecture = [2]

# trained_ann = MLP.MLP.train(x=x,
#                             y=y,
#                             net_arch=network_architecture,
#                             max_iter=30000,
#                             learning_rate=0.01,
#                             activation="relu", # sigmoid does not work...
#                             # GD_type="batch",  # stochastic or batch
#                             debug=True)

# last example of the book modified to work
trained_ann = MLP.MLP.train(x=x,
                            y=y,
                            net_arch=network_architecture,
                            max_iter=200000,
                            learning_rate=0.01,
                            activation="relu", # sigmoid does not work...
                            GD_type="batch",  # stochastic or batch
                            debug=False)


print("\nTraining Time : ", trained_ann["training_time_sec"])
print("Number of Training Iterations : ", trained_ann["elapsed_iter"])
print("Network Architecture : ", trained_ann["net_arch"])
print("Network Error : ", trained_ann["network_error"])

predicted_output = MLP.MLP.predict(trained_ann, x)
print("\nPredicted Output(s) : ", predicted_output)
