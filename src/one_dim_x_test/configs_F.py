#! /usr/bin/python3

##-------------------------------------------------------------------------------\
#   surrogate_model_optimization
#   './surrogate_model_optimization/src/one_dim_x_test/configs_F.py'
#   configurations for function compatable with project optimizers
#
#   Author(s): Lauren Linkous (LINKOUSLC@vcu.edu)
#   Last update: June 26, 2025
##-------------------------------------------------------------------------------\


import sys

try: # for outside func calls
    sys.path.insert(0, './pso_basic_multi_glods_surrogate/src/')
    from one_dim_x_test.func_F import func_F
    from one_dim_x_test.constr_F import constr_F
except: # for local
    from func_F import func_F
    from constr_F import constr_F

OBJECTIVE_FUNC = func_F
CONSTR_FUNC = constr_F
OBJECTIVE_FUNC_NAME = "one_dim_x_test.func_F"
CONSTR_FUNC_NAME = "one_dim_x_test.constr_F"

# problem dependent variables
LB = [[0]]             # Lower boundaries
UB = [[1]]               # Upper boundaries
IN_VARS = 1                 # Number of input variables (x-values)
OUT_VARS = 1                # Number of output variables (y-values) 
TARGETS = [0]               # Target values for output
GLOBAL_MIN = [[0.974857, -0.954872]]       # Global minima sample, if they exist. 