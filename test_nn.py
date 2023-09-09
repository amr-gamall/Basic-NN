from  nn_model import nn_model as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('tvmarketing.csv')
input, output = [np.array(df.TV), np.array(df.Sales)]

n = len(input)
input = input.reshape((1, len(input)))
output = output.reshape((1, len(output)))

test = nn(input, output, n, 1.2, 10000)

tst_input = np.linspace(0, 100)
tst_input = tst_input.reshape((1, tst_input.shape[0]))

test.learn()

print(test.get_prediction(tst_input))

