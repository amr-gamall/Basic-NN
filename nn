import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from init_nn import init_nn
class nn_model:
    def __init__(self, data_input, data_output, input_size, learning_rate, iterations):
        self.iterations = iterations
        self.init_network = init_nn(data_input, data_output, input_size, learning_rate)
        self.init_network.normalize_parameters()
    def learn(self):
        for i in range(self.iterations):
            self.init_network.back_prop()
    def get_prediction(self, input_to_predict):
        return self.init_network.predict(input_to_predict)





