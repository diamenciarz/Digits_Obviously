import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def extract_weights(nn_params, input_layer_size, hidden_layer_size, num_labels):
    tmp = nn_params.copy()
    # Take first 401*25 values and put the m into an 401x25 grid
    division_point = hidden_layer_size * (input_layer_size + 1)
    Theta1 = np.reshape(
        tmp[0:division_point], (hidden_layer_size, (input_layer_size + 1)), order='F')
    # Take second 26*10 values and put the m into an 26*10 grid
    Theta2 = np.reshape(tmp[division_point:len(tmp)],
                        (num_labels, (hidden_layer_size + 1)), order='F')
    return Theta1, Theta2


def forward_propagation(training_instance_count, Theta1, Theta2, X):
    a2 = sigmoid(np.dot(
        np.hstack((np.ones((training_instance_count, 1)), X)), np.transpose(Theta1)))
    a3 = sigmoid(np.dot(
        np.hstack((np.ones((training_instance_count, 1)), a2)), np.transpose(Theta2)))

    return a2, a3


def calculate_activations(X, Theta1, Theta2, t):
    a1 = append_one(X[t])

    z2 = np.dot(a1, Theta1.T)
    a2 = append_one(sigmoid(z2))

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return a3, z2, a2, a1


def append_one(z):
    return np.append(1, z)


def one_hot(y, classes):
    array = np.zeros([classes])
    # If 3 classes, then 3rd is at index 2
    array[y - 1] = 1
    return array


def list_of_elements(array):
    # Adds one dimension to the array, putting each element in its separate list
    return np.array([[i] for i in array])


def gradient2(a2, dz3):
    p1 = list_of_elements(a2)
    p2 = list_of_elements(dz3)
    return np.dot(p1, p2.T)


def gradient1(dz2, a1):
    p1 = list_of_elements(a1)
    return np.dot(dz2, p1.T)


def calculate_gradients(a3, z2, a2, a1, t, y):
    # We subtract
    num_of_classes = a3.shape[0]
    dz3 = a3 - one_hot(y[t], num_of_classes)

    dw2 = gradient2(a2, dz3)
    # Leave out the error or the previous bias, the bias of weights2 will already be affected
    # Add a dimension to dz3, so that we can take the dot product of this matrix
    sg2 = list_of_elements(sigmoidGradient(z2))
    dz2 = np.dot(dw2[1:], list_of_elements(dz3)) * sg2

    dw1 = gradient1(dz2, a1)
    return dw1, dw2.T


def backpropagate(X, y, Theta1, Theta2, t):
    # X are the features of each training instance
    # y are the correct classes
    # t is the index of the instance that we are backpropagating over
    a3, z2, a2, a1 = calculate_activations(X, Theta1, Theta2, t)
    dw1, dw2 = calculate_gradients(a3, z2, a2, a1, t, y)
    return dw1, dw2


def accumulate_grad(X, y, Theta1, Theta2):
    accumulated_grad1 = np.zeros(np.shape(Theta1))
    accumulated_grad2 = np.zeros(np.shape(Theta2))
    for t in range(X.shape[0]):
        # here we calculate the modified values of the new weights for both layers
        dw1, dw2 = backpropagate(X, y, Theta1, Theta2, t)
        accumulated_grad1 += dw1
        accumulated_grad2 += dw2
    return accumulated_grad1, accumulated_grad2

def zero_first_column(theta):
    for i in range(theta.shape[0]):
        theta[i,0] = 0
    return theta

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, learning_rate):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)
    #   computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1, Theta2 = extract_weights(
        nn_params, input_layer_size, hidden_layer_size, num_labels)

# Setup some useful variables
    # 5000 instances
    training_instance_count = np.shape(X)[0]

# Computation of the Cost function including regularisation
# Feedforward
    a2, a3 = forward_propagation(training_instance_count, Theta1, Theta2, X)

    # Cost function for Logistic Regression summed over all output nodes
    cost_per_class = np.empty((num_labels, 1))
    for t in range(num_labels):
        # Array with True, where class matches (k + 1)
        y_binary = (y == t+1)
        # Array that is the k-th column of a3. a3 has all the weighted class predictions
        # hk has all the prediction weights for a specific class
        hk = a3[:, t]
        # compute two parts of cost function for all examples for node k
        cost_per_class[t][0] = np.sum(np.transpose(
            y_binary)*np.log(hk)) + np.sum(((1-np.transpose(y_binary))*np.log(1-hk)))

# Sum over all labels and average over examples
    J_no_regularisation = -1./training_instance_count * sum(cost_per_class)
# No regularization over intercept
    Theta1_no_intercept = Theta1[:, 1:]
    Theta2_no_intercept = Theta2[:, 1:]

# Sum all parameters squared
    RegSum1 = np.sum(np.sum(np.power(Theta1_no_intercept, 2)))
    RegSum2 = np.sum(np.sum(np.power(Theta2_no_intercept, 2)))
# Add regularisation term to final cost
    J = J_no_regularisation + \
        (learning_rate/(2*training_instance_count)) * (RegSum1+RegSum2)

# You need to return the following variables correctly
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

# ====================== YOUR CODE HERE ======================
# Implement the backpropagation algorithm to compute the gradients
# Theta1_grad and Theta2_grad. You should return the partial derivatives of
# the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# Theta2_grad, respectively. After implementing Part 2, you can check
# that your implementation is correct by running checkNNGradients
#
# Note: The vector y passed into the function is a vector of labels
#       containing values from 1..K. You need to map this vector into a
#       binary vector of 1's and 0's to be used with the neural network
#       cost function.
#
# Hint: It is recommended implementing backpropagation using a for-loop
#       over the training examples if you are implementing it for the
#       first time.
#



    Theta1_grad, Theta2_grad = accumulate_grad(X, y, Theta1, Theta2)
    Theta1_grad /= training_instance_count
    Theta2_grad /= training_instance_count 
    # Regularization
    Theta1_grad += (learning_rate / training_instance_count) * zero_first_column(Theta1)
    Theta2_grad += (learning_rate / training_instance_count) * zero_first_column(Theta2)

# -------------------------------------------------------------

# =========================================================================

# Unroll gradients
    Theta1_grad = np.reshape(Theta1_grad, Theta1_grad.size, order='F')
    Theta2_grad = np.reshape(Theta2_grad, Theta2_grad.size, order='F')
    grad = np.expand_dims(np.hstack((Theta1_grad, Theta2_grad)), axis=1)

    return J, grad
