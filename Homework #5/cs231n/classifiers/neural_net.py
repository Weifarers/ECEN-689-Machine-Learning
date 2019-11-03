from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # This is only the first layer of the forward pass, which is a fully connected layer with
        # a ReLU. 
        
        # Since it's a fully connected layer, we just take the dot product of X and W1, and add the bias.
        layer_1 = X.dot(W1) + b1
        # A ReLU is just a maximum comparing to 0. 
        layer_out_1 = np.maximum(0, layer_1)
        # Then we calculate the scores from the first layer. These are the inputs to the second layer.
        scores = np.dot(layer_out_1, W2) + b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # After the second layer, we have a Softmax classifier. In this case, since we've
        # already done the input for the second layer above, we just need to start with
        # the softmax calculations. We'll use the vectorized version for speed.
        scores_exp = np.exp(scores)
        # Like in Linear SVM, we fix matrix sizes to avoid Python yelling at us.
        scores_sum = np.sum(scores_exp, axis=1)
        scores_sum = scores_sum.reshape(N,1)
        # Then, we can directly calculate the probabilities.
        prob = scores_exp/scores_sum
        
        # Now we can calculate the losses. Doing a summation of the negative is the same thing
        # as constantly subtracting. Note that we only take the probabilities where the numerator
        # is the correct class y, so we have to look over every row and choose the corresponding 
        # column y.
        loss = np.sum(-np.log(prob[np.arange(N), y]))
        
        # Dividing to get averages, and adding regularization.
        loss /= N
        # Since there's two sets of weights, we need to consider both W1 and W2 when doing 
        # regularization. 
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # We start with the probabilities, which are the scores of each class. 
        d_prob = prob
        # We then take the derivative of d_prob, with respect to the inputs to the second layer.
        # For all the correct elements, like when we're doing the gradient of softmax, we need 
        # to subtract 1. This is part of the definition for the gradient of softmax.
        d_prob[np.arange(N), y] -= 1
        # Since we're actually considering the derivative of the loss with respect to our 
        # weights and biases, the loss function has a constant 1/N on the outside, so we 
        # multiply our d_prob by the same.
        d_prob /= N
        
        # But, we want the derivative of the scores with respect to the weights of the second 
        # layer, W2. By chain rule, if S is the score:
        # dS/dW2 = dS/dU * dU/dW2, where U is the input to the second layer. 
        # We've already done dS/dU, and dU/dW2 is actually just the output of the first layer.
        grads['W2'] = np.dot(layer_out_1.T, d_prob)
        # Don't forget regularization! 
        grads['W2'] += 2 * reg * W2
        # The same principle applies when considering the biases. 
        # dS/dB2 = dS/dU * dU/dB2, but dU/dB2 is just a bunch of 1's, so we're really just summing
        # all the elements of dS/dU.
        grads['b2'] = np.sum(d_prob, axis=0)
        
        # Now, we're backpropagating into the first layer. As an example, let's say we're looking at
        # dS/dW1. dS/dW1 = dS/dU2 * dU2/dy1 * dy1/dU1 * dU1/dW1. We already have dS/dU2 with d_prob, 
        # and dU2/dy1 is just the weights W2. We use the transpose for matching matrix dimensions.
        d_relu = np.dot(d_prob, W2.T)
        # dy1/dU1 is the derivative of the RELU function. The derivative of the RELU with respect to 
        # it's inputs is actually just 1, but for any input that was 0, the derivative is 0. So, we 
        # check d_relu for any outputs that resulted in 0, and set those equal to 0. 
        d_relu[layer_out_1 <= 0] = 0
        # Finally, the gradients with respsect to the weights W1 and b1 are similar in definition to
        # those of W2 and b2. The only difference is now that we're at the first layer, we're 
        # multiplying by X, the original inputs.
        grads['W1'] = np.dot(X.T, d_relu)
        # Same as for W2, including regularization.
        grads['W1'] += 2 * reg * W1
        grads['b1'] = np.sum(d_relu, axis=0)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Randomly select batches based on batch_size, with replacement.
            temp_batch = np.random.choice(num_train, batch_size, replace=True)
            
            # Gets the corresponding values for X and y. 
            X_batch = X[temp_batch]
            y_batch = y[temp_batch]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # In stochastic gradient descent, our learning rate is our eta, and we use grad for
            # the gradient.
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # First, we get the input to the first layer, and pass it trhough RELU.
        layer_in_1 = np.dot(X, self.params['W1']) + self.params['b1']
        layer_out_1 = np.maximum(0, layer_in_1)
        # The associated scores are the outputs of the RELU layer multiplied by weights, with biases. 
        scores = np.dot(layer_out_1, self.params['W2']) + self.params['b2']
        # We select the classes that correspond to the highest score. 
        y_pred = np.argmax(scores, axis=1)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
