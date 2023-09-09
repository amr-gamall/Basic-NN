import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class init_nn:
    def __init__(self, data_input, data_output, size, lr):
        self.mean_input = np.array(data_input).mean()
        self.mean_output = np.array(data_output).mean()

        self.std_input = np.array(data_input).std()
        self.std_output = np.array(data_output).std()

        self.learning_rate = lr
        self.prediction = np.ones((data_output.shape))

        self.data_size = size
        self.data_input = data_input
        self.data_output = data_output
        
        self.num_inputs = data_input.shape[0]
        self.num_outputs = data_output.shape[0]

        self.dim_weight = (self.num_outputs, self.num_inputs)
        self.dim_bias = data_output.shape[0]

        self.weight = np.ones(self.dim_weight)
        self.bias = np.ones(self.dim_bias)

    def normalize_parameters(self):

        self.data_input = (self.data_input - np.array(self.data_input).mean())/np.array(self.data_input).std()
        self.data_output = (self.data_output - np.array(self.data_output).mean())/np.array(self.data_output).std()

    def forward_prop(self): # return array each elem is output for each dataset
        self.prediction = self.weight @ np.array(self.data_input) + self.bias
    

    def cost(self): # return scalar aver total cost
        n = self.data_size
        return  1/(2*n) * np.sum((self.prediction - self.data_output) ** 2, axis = 1)
    
    def dCost_dW(self): # return array of change of cost wrt to every weight
        n = self.data_size
        return 1/n * ((self.prediction - self.data_output)) @ self.data_input.T

    def dCost_dB(self): # return array of change of cost wrt to every bias
        n = self.data_size
        return 1/n * np.sum(((self.prediction - self.data_output)), axis = 1,  keepdims = True)

    def back_prop(self):
        self.forward_prop()
        self.weight = self.weight - self.dCost_dW() * self.learning_rate
        self.bias = self.bias - self.dCost_dB() * self.learning_rate            


    def denormalize(self, to_denormalize):
        return to_denormalize * self.std_output + self.mean_output


    def predict(self, in_to_predict):
        in_to_predict = (in_to_predict - self.mean_input)/self.mean_input
        return self.denormalize(self.weight @ np.array(in_to_predict) + self.bias)


if __name__ == "__main__":

    df = pd.read_csv('tvmarketing.csv')
    input, output = [np.array(df.TV), np.array(df.Sales)]
    n = len(input)
    input = input.reshape((1, len(input)))
    output = output.reshape((1, len(output)))
    x = init_nn(input, output, n, 0.01)
    x.normalize_parameters()
    # x.back_prop()
    # print(x.weight @ [10] + x.bias)
    # print(x.predict([10]))



