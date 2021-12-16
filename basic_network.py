# coding: utf-8
input_vector= [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2= [2.17, 0.32]

#computing the dot of input_vector and weighs_1
first_indexes_mult= input_vector[0] * weights_1[0]
second_indexes_mult= input_vector[1] * weights_1[1]
dot_product_1= first_indexes_mult + second_indexes_mult
print(f"The dot product is: {dot_product_1}")

import numpy as np

dot_product_1 = np.dot(input_vector, weights_1)
print(dot_product_1)
dot_product_2= np.dot(input_vector, weights_2)
print(dot_product_2)

#wrapping the vectors in numpy arrays
input_vector= np.array([1.66, 1.56])
weights_1= np.array([1.45, -0.66])
bias= np.array([0.0])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1= np.dot(input_vector, weights)+ bias
    layer_2= sigmoid(layer_1)
    return layer_2

presdiction= make_prediction(input_vector, weights_1, bias)
prediction = make_prediction(input_vector, weights_1, bias)

print(f"prediction is: {prediction}")

input_vector= np.array([2, 1.5])
prediction= make_prediction(input_vector, weights_1, bias)

print(prediction)

input_vector= np.array([1, 1.8])
prediction= make_prediction(input_vector, weights_1, bias)

print(prediction)

input_vector= np.array([.5, 2.5])

print(make_prediction(input_vector, weights_1, bias))

target = 0
mse= np.square(prediction - target)

print(f'prediction:{prediction}; error: {mse}')
derivative= 2*(prediction- target)
print(f'the derivative is: {derivative}')

#updating the weights
weights_1= weights_1- derivative
prediction= make_prediction(input_vector, weights_1, bias)
error= (prediction- target)**2

print(f'prediction: {prediction}; error {error}')

def sigmoid_deriv(x):sigmoid(x)* (1-sigmoid(x))

def sigmoid_deriv(x):
    return sigmoid(x)* (1-sigmoid(x))

derror_dprediction= 2* (prediction-target)
layer_1= np.dot(input_vector, weights_1)+ bias
dprediction_dlayer1= sigmoid_deriv(layer_1)
dlayer1_dbias= 1

derror_dbias= (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    def train(self, input_vectors, targets, iterations):
        cumulative_errors=[]
        for currents_iteration in range(iterations):
            #pick a data instance at random
            random_data_index= np.random.randint(len(input_vectors))
            input_vector= input_vectors[random_data_index]
            target= targets[random_data_index]
            #compute the gradients and update the weights_1
            derror_dbias, derror_dweights= self._compute_gradients(
                input_vector, targets
            )

            self._update_parameters(derror_dbiass, derror_dweights)

            # measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                #loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point= input_vectors[data_instance_index]
                    target= targets[data_instance_index]

                    prediction= self.predict(data_point)
                    error= np.square(prediction - target)

                    cumulative_error= cumulative_error + error

                cumulative_errors.append(cumulative_error)
        return cumulative_errors



learning_rate= 0.1
neural_network= NeuralNetwork(learning_rate)

print(neural_network.predict(input_vector))
