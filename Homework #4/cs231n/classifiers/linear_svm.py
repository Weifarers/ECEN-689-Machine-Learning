from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    # Gets the number of classes, and the amount of training data.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # Initializes the losses.
    loss = 0.0
    # Iterates through all the training data.
    for i in range(num_train):
        # Gets the associated score of each image by multiplying it by the weights.
        # Each possible class has an associated score. 
        scores = X[i].dot(W)
        # Gets the score associated with the correct class. 
        correct_class_score = scores[y[i]]
        # Goes through each potential class.
        for j in range(num_classes):
            # If we're at the true class, we skip. 
            if j == y[i]:
                continue
            # Calculates the margin for all the incorrect classes.
            # The idea behind SVM is that we want the correct class to have
            # a higher score than the incorrect classes by some margin. 
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # Since our loss function is a linear function, its gradient with respsect to the
                # weights is just the sum of the constants, X[i]. 
                # This updates the column of the gradient for any incorrect class.
                dW[:,j] += X[i,:]
                # This updates the column of the gradient for the correct class. We do a subtraction 
                # here based on the equivalent form of the loss function, given right after the example.
                dW[:,y[i]] -= X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Taking the derivative of the regularization to add on. 
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # We can mimic the above, where we also take the amount of training data.
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # All we're doing here is vectorizing the SVM loss, meaning we should expect the same results,
    # but we have no for loops. 
    
    # The scores, defined as f(x,W), are just a dot product between the two vectors.
    scores = np.dot(X,W)
    
    # In this problem, we're trying to subtract the correct scores from the scores associated with
    # the correct class, along with a margin (+1). So to start, we need the exact indices of every
    # correct score in the scores matrix.
    
    # Since we already have the locations of all the correct scores, inside y, we can just use 
    # numpy's choose function to take them out. The size of y dictates the number of columns for 
    # the matrix that we're choosing from, so we have to transpose it.
    correct_scores = np.choose(y, np.transpose(scores))
    
    # After getting the correct scores, we can just subtract them from the original scores, and add 1.
    # But, we need to enforce the size of correct_scores so that we don't run into any matrix manipulation
    # errors. 
    correct_scores = correct_scores.reshape(num_train,1)
    # Then we calculate the margin, which is defined in our for loop as the scores minus the correct scores
    # plus one.
    margin = scores - correct_scores + 1
    
    # Finally, we need to adjust all the elements that correspond to the correct classes, and set them 
    # equal to 0. (They're currently at value 1 since we added delta.) We just check wherever the margin
    # is equal to exactly 1, and replace it with 0. Otherwise, it retains its original value. 
    margin = np.where(margin == 1, 0, margin)
    margin = np.maximum(margin, 0)
    
    # We can finally calculate the loss, which is just the sum of all of the margins, averaged by the amount
    # of training data. We convert the amount of training data to a float so we can get decimal values.
    loss = np.sum(margin) / float(num_train)
    
    # Tacking on regularization too.
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # To calculate the gradient, we can use the margin, except we have to avoid all the values
    # that correspond to the correct class when we're doing summations, and do subtractions for 
    # them instead.
    # First, we'll set all values where the margin is greater than 0 (incorrect classes), and set them
    # equal to 1. Then we force the other values to be 0. (They're technically already 0, but Python is picky
    # about its syntax.)
    incorrect = np.where(margin > 0, 1, 0)
    # Equivalently, for every instance in which there is an incorrect class, we also subtract the margin
    # from the correct class's margin. Some rows have more than one correct class, so we can't just 
    # arbitrarily throw around -9s, unfortunately.
    # Thus, we find all the indices of the correct classes marked by y, and set them equal to the 
    # summation of the corresponding rows, multiplied by -1.
    incorrect[range(num_train), y] = -np.sum(incorrect, axis=1)
    
    # Finally, we can calculate the gradient, which we take the average of.
    dW = np.dot(X.T, incorrect) / float(num_train)
    
    # Tacking on regularization too.
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
