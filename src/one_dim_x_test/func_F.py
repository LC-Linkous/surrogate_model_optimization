#! /usr/bin/python3

##-------------------------------------------------------------------------------\
#   surrogate_model_optimization
#   './surrogate_model_optimization/src/one_dim_x_test/func_F.py'
#   objective function for function compatable with project optimizers
#
#   Author(s): Lauren Linkous (LINKOUSLC@vcu.edu)
#   Last update: June 26, 2025
##-------------------------------------------------------------------------------\

import numpy as np
import time

def func_F(X, NO_OF_OUTS=1):
    F = np.zeros((NO_OF_OUTS))
    noErrors = True
    try:
        x = X[0]
        F = np.sin(5 * x**3) + np.cos(5 * x) * (1 - np.tanh(x ** 2))
    except Exception as e:
        print(e)
        # print("X!")
        # print(X)
        noErrors = False

    return [F], noErrors
