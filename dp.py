import numpy as np

# ----------------------------
# 1) Network structure
# ----------------------------
n = 2                      # number of inputs
num_hidden_layers = 2
m = [2, 2]                 # nodes in each hidden layer
num_nodes_output = 1

np.random.seed(12)         # reproducible randomness

# ----------------------------
# 2) Initialize network
# ----------------------------
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = "output"
            num_nodes = num_nodes_output
        else:
            layer_name = f"layer_{layer+1}"
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f"node_{node+1}"
            network[layer_name][node_name] = {
                "weights": np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                "bias": float(np.around(np.random.uniform(size=1), decimals=2)[0])  # store as float
            }

        num_nodes_previous = num_nodes

    return network

small_network = initialize_network(n, num_hidden_layers, m, num_nodes_output)
print("Network:\n", small_network)

# ----------------------------
# 3) Math helpers
# ----------------------------
def compute_weighted_sum(inputs, weights, bias):
    inputs = np.array(inputs, dtype=float)
    weights = np.array(weights, dtype=float)
    return np.sum(inputs * weights) + bias

def node_activation(weighted_sum):
    # sigmoid
    return 1.0 / (1.0 + np.exp(-weighted_sum))

# ----------------------------
# 4) Forward propagation
# ----------------------------
def forward_propagate(network, inputs):
    layer_inputs = list(inputs)

    for layer_name, layer_data in network.items():
        layer_outputs = []

        for node_name, node_data in layer_data.items():
            z = compute_weighted_sum(layer_inputs, node_data["weights"], node_data["bias"])
            a = node_activation(z)
            layer_outputs.append(float(np.around(a, decimals=4)))

        if layer_name != "output":
            print(f"Outputs of {layer_name}: {layer_outputs}")

        layer_inputs = layer_outputs

    return layer_outputs  # final output layer

# ----------------------------
# 5) Test input (must match n=2)
# ----------------------------
inputs = np.around(np.random.uniform(size=n), decimals=2)
print("\nInputs:", inputs)

predictions = forward_propagate(small_network, inputs)
print("\nPrediction:", predictions[0])
