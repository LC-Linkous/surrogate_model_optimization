#! /usr/bin/python3

##--------------------------------------------------------------------\
#   surrogate_model_optimization
#   './surrogate_model_optimization/src/lundquist_3_var/constr_default.py'
#   Function for default constraints. Called if user does not pass in 
#       constraints for objective function or problem being optimized. 
#
#
#   Author(s): Jonathan Lundquist, Lauren Linkous 
#   Last update: June 26, 2025
##--------------------------------------------------------------------\



import numpy as np

def constr_default(X):
    return True