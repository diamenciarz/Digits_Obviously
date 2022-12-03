import numpy as np

from sigmoid import sigmoid

def calculate_activations(theta, X):
    # Calculate activations
    z = [0] * theta.shape[0]
    for node in range(theta.shape[0]):
        # Add the bias
        z[node] += theta[node,0]
        for i in range(1,theta.shape[1]):
            z[node] += X[i - 1] * theta[node,i]

    return z

def make_column_vector(vector):
    return np.array([vector]).T

def choose_class(final_activations):
    highest_value = float('-inf')
    highest_index = -1
    for i in range(len(final_activations)):
        if final_activations[i] > highest_value:
            highest_index = i
            highest_value = final_activations[i]
    return highest_index + 1

def predict(Theta1, Theta2, Xs):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

    p = np.zeros([len(Xs),1])
    for i in range(len(Xs)):
        z2 = calculate_activations(Theta1, Xs[i])
        z3 = calculate_activations(Theta2, z2)
        p[i] = choose_class(z3)
    return np.squeeze(make_column_vector(p))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#

# =========================================================================
