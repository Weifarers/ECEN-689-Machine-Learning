from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Just like in linear SVM, we start by getting the number of classes, and amount of training data.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        # Get the scores in the same manner as Linear SVM.
        scores = X[i].dot(W)
        # Then we do element-wise exponetials. 
        scores_exp = np.exp(scores)
        
        # The correct score is found for each of our training data, which is the numerator.
        correct_exp = scores_exp[y[i]]
        # Then we find the probability.
        temp_prob = correct_exp / np.sum(scores_exp)
        # Take the logarithm of the probability, which we use to calculate the loss. 
        loss -= np.log(temp_prob)
        
        # Now, for each of the classes, we need to update the gradients for each training data point.
        for k in range(num_classes):
            # There's a few parts to the gradient function: 
            # - There's a -1(y_i == k)
            # - There's also a e^(f_k)/np.sum(scores_exp)
            # - Finally, since there's a derivation with respect to the weights, all of this is multiplied
            # by X[i]. 
            dW[:,k] += ((scores_exp[k] / np.sum(scores_exp)) - (y[i] == k)) * X[i]
   
    # Just like in linear SVM, we divide by the number of training data to get averages, and add reguliarization.
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    # We can mimic the above, where we also take the amount of training data.
    num_train = X.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Again, we're looking for the same exact results, implemented in a faster way. 
    
    # Begin by calculating all the scores, and taking the exponent of them.
    scores = X.dot(W)
    scores_exp = np.exp(scores)
    # Like in Linear SVM, we fix matrix sizes to avoid Python yelling at us. 
    scores_sum = np.sum(scores_exp, axis=1)
    scores_sum = scores_sum.reshape(num_train,1)
    # Then, we can directly calculate the probabilities.
    prob = scores_exp/scores_sum
    
    # Now we can calculate the losses. Doing a summation of the negative is the same thing
    # as constantly subtracting. Note that we only take the probabilities where the numerator
    # is the correct class y, so we have to look over every row and choose the corresponding 
    # column y. 
    loss = np.sum(-np.log(prob[np.arange(num_train), y]))
    
    # When calculating the gradient, we need a matrix that contains a bunch of 1's, wherever 
    # y_i == k. We can actually do this in the same manner as above, where we just generate 
    # a matrix of a bunch of zeros, and every column that corresponds to the correct class of 
    # the associated row just becomes one.
    yi_k_matrix = np.zeros_like(prob)
    yi_k_matrix[np.arange(num_train), y] = 1
    
    # Then we take the entire probability matrix, subtract yi_k_matrix from it, and dot it with
    # the X matrix. 
    dW = np.dot(X.T, prob - yi_k_matrix)
    
    # Just like in linear SVM, we divide by the number of training data to get averages, and add reguliarization.
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
