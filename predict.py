import numpy as np

from sigmoid import sigmoid

def calculate_activations(theta, X):
    # Calculate activations
    z2 = [0] * theta.shape[0]
    for node in range(theta.shape[0]):
        # Add the bias
        z2[node] += theta[node,0]
        for i in range(1,theta.shape[1]):
            z2[node] += X[i - 1] * theta[node,i]

    return z2

def make_column_vector(vector):
    return np.array([vector]).T

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

    
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
    z2 = calculate_activations(Theta1, X)
    z3 = calculate_activations(Theta2, z2)
    return make_column_vector(z3)

# =========================================================================
