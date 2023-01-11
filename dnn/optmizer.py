class Optimizer:
    def __init__(self, propagator, loss, learning_rate, predictor):
        self.propagator = propagator
        self.loss = loss
        self.learning_rate = learning_rate
        self.predictor = predictor

    
    def gradient_descent(self, model, iters):
        """
        Trains the model using gradient descent.
    
        Arguments:
            - model: the model to be trained
            - iters: the number of iterations to train the model
    """
        params = model.params
        inputs = model.inputs
        outputs = model.outputs
        outputs_encoded = utils.one_hot_encode(outputs)

        L = len(params_in) // 2

        accuracies = []
        losses = []

        for it in range(1, iters + 1):
            # compute activations
            activations = self.propagator.forward(inputs, params)

            # make predictions (right-most set of activations)
            Y_hat = self.predictor.max_activation(activations["A"+str(L)])

            # compute accuracy
            accuracy = model.metrics.get_accuracy(Y_hat, outputs)
            accuracies.append(accuracy)

            # compute loss
            loss = model.metrics.cross_entropy(outputs_encoded, activations["A"+str(L)])
            losses.append(loss)

            # compute gradients
            grads = self.propagator.backward(activations, params, outputs)

            # update parameters
            params = self.update_params(params, grads, self.learning_rate)
            model.params = params

            if it % 100 == 0:
                print("Iteration: ", it, "Loss: ", loss, "Accuracy: ", accuracy)

        return model, accuracies, losses


    def update_params(self, params, grads, learning_rate):
        """
        Updates the parameters of the network using gradient descent.
            
        Arguments:
            - params: dictionary containing the parameters W and b for each layer
            - grads: dictionary containing the gradients dW and db for each layer
            - alpha: learning rate
    
        Returns:
            - params_updated: dictionary containing the updated parameters W and b for each layer
        """
        # number of layers
        L = len(params) // 2
        
        params_updated = {}
        for l in range(1, L+1):
            params["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
            params["b"+str(l)] -= learning_rate * grads["db"+str(l)]

        return params_updated
