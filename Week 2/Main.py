import numpy as np
from matplotlib import pyplot
from MultiLayerPerceptron import MultiLayerPerceptron

# we have a vectorized version of the function y = xˆ3 - xˆ2 + 1
target_function = np.vectorize(lambda x: x ** 3 - x ** 2 + 1)

# generate 100 random inputs
inputs = np.random.random_sample(100)
# calculate output of the function for the inputs, which will be out targets
targets = target_function(inputs)

# # plot inputs vs target
# pyplot.scatter(inputs, targets)
# pyplot.xlabel("x")
# pyplot.ylabel("t")
# pyplot.title("t = x^3 - x^2 + 1")
# pyplot.show()

learning_rate = 0.02
mlp = MultiLayerPerceptron()

loss = []
for epoch in range(0, 100):
    total_loss_in_epoch = 0
    for i in range(0, inputs.size):
        mlp.forward_step([[inputs[i]]], [[targets[i]]])
        print(f"loss for input {i} in epoch {epoch}: {mlp.output_layer.loss}")
        total_loss_in_epoch += mlp.output_layer.loss[0][0]
        mlp.backpropagation(learning_rate)
    average_loss_in_epoch = total_loss_in_epoch / inputs.size
    print(f"average loss in epoch: {average_loss_in_epoch}")
    loss.append(average_loss_in_epoch)
print(loss)
pyplot.plot(loss)
pyplot.xlabel("epoch")
pyplot.ylabel("avg loss")
pyplot.title("loss over epochs")
pyplot.show()
