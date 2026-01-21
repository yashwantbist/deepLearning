## Objective for this Notebook
## Build a Neural Network
## Compute Weighted Sum at Each Node
## Compute Node Activation
## Use Forward Propagation to Propagate Data 


#install #!pip install numpy==1.26.4


#import numpy library
import numpy as np

#we initializes the weights and biases
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)

#print the weights and biases for check
print(weights)
print(biases)

#now, we have the weights and the biases defined for the network, lets compute for a 
#given input, x1, x2.
x_1 = 0.5
x_2 = 0.82

print('x1 is {} and x2 is {}'.format(x_1, x_2))

#lets start by computing the weighted sum of the inputs, z11 at thefirst node of the hidden layer.
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

#compute the weighted sum if the inputs, z12 at second node
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_12))

#calculating a sigmond function, lets compute the activation of first node a11
a_11 = 1.0/(1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# Calcuate a sigmond function, for second node a_12
a_12 = 1.0/(1.0 + np.exp(-z_12))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

# Now these activations will serve as the inputs to the output layer.
# So, let's compute the weighted sum of these inputs to the node in the output layer. 
# Assign the value to **z_2**.

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))


#Finally, let's compute the output of the network as the activation of 
# the node in the output layer. Assign the value to a_2.
# create a sigmond function
a_2 = 1.0 / (1.0 + np.exp(-z_2))

print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))