import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def extract_weights(nn_params, input_layer_size, hidden_layer_size, num_labels):
    tmp = nn_params.copy()
    # Take first 401*25 values and put the m into an 401x25 grid
    division_point =hidden_layer_size * (input_layer_size + 1)
    Theta1 = np.reshape(tmp[0:division_point], (hidden_layer_size, (input_layer_size + 1)), order='F')
    # Take second 26*10 values and put the m into an 26*10 grid
    Theta2 = np.reshape(tmp[division_point:len(tmp)], (num_labels, (hidden_layer_size + 1)), order='F')
    return Theta1, Theta2

def forward_propagation(training_instance_count, Theta1, Theta2, X):
    a2 = sigmoid(np.dot(np.hstack((np.ones((training_instance_count, 1)), X)), np.transpose(Theta1)))
    a3 = sigmoid(np.dot(np.hstack((np.ones((training_instance_count, 1)), a2)), np.transpose(Theta2)))
    
    return a2, a3

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value):
#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
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
    Theta1, Theta2 =extract_weights(nn_params, input_layer_size, hidden_layer_size, num_labels)

# Setup some useful variables
    # 5000 instances
    training_instance_count = np.shape(X)[0]

# Computation of the Cost function including regularisation
# Feedforward 
    a2, a3 = forward_propagation(training_instance_count, Theta1, Theta2, X)

    # Cost function for Logistic Regression summed over all output nodes
    cost_per_class = np.empty((num_labels, 1))
    for k in range(num_labels):
        # Array with True, where class matches (k + 1)
        y_binary=(y==k+1)
        # Array that is the k-th column of a3. a3 has all the weighted class predictions
        # hk has all the prediction weights for a specific class
        hk=a3[:,k]
        # compute two parts of cost function for all examples for node k
        cost_per_class[k][0] = np.sum(np.transpose(y_binary)*np.log(hk)) + np.sum(((1-np.transpose(y_binary))*np.log(1-hk)))
        
# Sum over all labels and average over examples
    J_no_regularisation = -1./training_instance_count * sum(cost_per_class)
# No regularization over intercept
    Theta1_no_intercept = Theta1[:, 1:]
    Theta2_no_intercept = Theta2[:, 1:]

# Sum all parameters squared
    RegSum1 = np.sum(np.sum(np.power(Theta1_no_intercept, 2)))
    RegSum2 = np.sum(np.sum(np.power(Theta2_no_intercept, 2)))
# Add regularisation term to final cost
    J = J_no_regularisation + (lambda_value/(2*training_instance_count)) * (RegSum1+RegSum2)

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





# -------------------------------------------------------------

# =========================================================================

# Unroll gradients
    Theta1_grad = np.reshape(Theta1_grad, Theta1_grad.size, order='F')
    Theta2_grad = np.reshape(Theta2_grad, Theta2_grad.size, order='F')
    grad = np.expand_dims(np.hstack((Theta1_grad, Theta2_grad)), axis=1)
    
    return J, grad
