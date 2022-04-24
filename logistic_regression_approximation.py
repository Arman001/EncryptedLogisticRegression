# Created by Muhammad Saad #
# On 04/23/22 #
# 4:00 PM     #
# ............ #

import numpy as np
from math import log



# It is not that difficult because it straight forward just multiply with weights and then pass through sigmoid #

class logistic:
    def __init__(self, no_of_inputs):
        self.no_of_input = no_of_inputs
        self.weights = np.random.rand(no_of_inputs)
        self.bias = np.random.rand(1)

    def cost_function(self, y, z):
        z_sig = self.deg3_sigmoid(z)                 
        predict_1 = y * log(z_sig)
        predict_0 = (1 - y) * log(1 - z_sig)
        return -(predict_1 + predict_0)

    def sigmoid(self,x):
        return 1/(1 + (np.exp(-x)))
    
    def deg3_sigmoid(self,x):
        return (0.5)+(1.20096*(x/8))-(0.81562*(x/8)**3)

    def training(self, input, y, lr, test):
        output = input * self.weights
        final_out = np.sum(output)
        final_out = final_out/255
        y_out = self.deg3_sigmoid(final_out) 
        loss = self.cost_function(y, y_out)  
        if(test==0):
            self.weights -= lr * (input*(y_out-y))
        return y_out, loss
