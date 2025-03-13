#! /usr/bin/python3

##--------------------------------------------------------------------\
#   surrogate_model_optimization
#   './surrogate_model_optimization/src/development_test.py'
#
#   This is the main development test class used to text compatibility 
#       between optimizer and estimator combinations. A system of commenting
#       out unused optimizer settings/declarations has been used to make 
#       swapping faster. This is NOT a clean, test/benchmarking file.
#
#   Benchmarking files are in labled by the base optimizer and 
#       surrogate model optimizer. Each combinaton has access to ALL 
#       functions used to fit the surrogate objective function.
#
#
#   Author(s): Lauren Linkous
#   Last update: February 18, 2024
##--------------------------------------------------------------------\

import time
import pandas as pd

# OPTIMIZER OPTIONS
from optimizers.surrogate_pso_python import swarm as pso_p_swarm
from optimizers.surrogate_pso_basic import swarm as pso_b_swarm
from optimizers.surrogate_pso_quantum import swarm as pso_q_swarm
from optimizers.surrogate_cat_swarm import swarm as catswarm
from optimizers.surrogate_sand_cat import swarm as sand_cat_swarm
from optimizers.surrogate_cat_quantum import swarm as cat_q_swarm
from optimizers.surrogate_chicken_swarm import swarm as chicken_swarm
from optimizers.surrogate_chicken_2015_improved import swarm as chicken_i15_swarm
from optimizers.surrogate_chicken_quantum import swarm as chicken_q_swarm
from optimizers.multi_glods.surrogate_multi_glods import multi_glods

# SURROGATE MODEL OBJECTIVE FUNCTION FIT OPTIONS
from surrogate_model.RBF_network import RBFNetwork
from surrogate_model.gaussian_process import GaussianProcess
from surrogate_model.kriging_regression import Kriging
from surrogate_model.polynomial_regression import PolynomialRegression
from surrogate_model.polynomial_chaos_expansion import PolynomialChaosExpansion
from surrogate_model.KNN_regression import KNNRegression
from surrogate_model.decision_tree_regression import DecisionTreeRegression
from surrogate_model.matern_process import MaternProcess
from surrogate_model.lagrangian_linear_regression import LagrangianLinearRegression
from surrogate_model.lagrangian_polynomial_regression import LagrangianPolynomialRegression

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
import himmelblau.configs_F as func_configs         # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs     # multi objective function


class MainTest():
    def __init__(self):


        # objective function values
        LB = func_configs.LB              # Lower boundaries, [[0.21, 0, 0.1]]
        UB = func_configs.UB              # Upper boundaries, [[1, 1, 0.5]]   
        IN_VARS = func_configs.IN_VARS    # Number of input variables (x-values)
        OUT_VARS = func_configs.OUT_VARS  # Number of output variables (y-values)
        TARGETS = func_configs.TARGETS    # Target values for output
        TOL = 10 **-4                   # Convergence Tolerance
        #TOL =  10 **-4                   # MultiGLODS Convergence Tolerance (This is a radius 
                                          #       based tolerance, not target based tolerance)
                                          #       Used by multi_glods
        MAXIT = 10000                     # Maximum allowed iterations
        self.sm_maxit = 5000              # Maximum allowed iterations for surrogate model
        self.sm_tol = 10 ** -4            # Convergence Tolerance for surrogate model
        self.out_vars = OUT_VARS #required for surrogate model call (for now)
        self.LB = LB
        self.UB = UB
        self.TARGETS = TARGETS


        # "ground truth" Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC  # objective function
        constr_F = func_configs.CONSTR_FUNC   # constraint function

        self.constr_F = constr_F


        '''
        BASE_OPT_CHOICE & SURROGATE_OPT_CHOICE  
        0: pso_python            1: pso_basic                     2: pso_quantum
        3: cat_swarm_python      4: sand_cat_python               5: cat_swarm_quantum
        6: chicken_swarm_python  7: 2015_improved_chicken_swarm   8: chicken_swarm_quantum
        9: multi_glods_python

        NOTE: multi_glods_python canNOT (currently) be used as a base optimizer with 
        a surrogate model optimizer. It can be used as the surrogate model optimizer.
        All models CAN be used stand-alone without a surrogate model 
        '''

        '''
        APPROXIMATOR_CHOICE
        0: RBF      1: Gaussian Process         2: Kriging       3:Polynomial Regression
        4: Polynomial Chaos Expansion  5: KNN regression   6: Decision Tree Regression
        7: Matern      8: Lagrangian Linear Regression  9:Lagrangian Polynomial Regression
        '''

        BASE_OPT_CHOICE = 1
        SURROGATE_OPT_CHOICE = 9
        APPROXIMATOR_CHOICE = 4
        
        # OPTIMIZER INIT
        self.best_eval = 10      #usually set to 1 because anything higher is too many magnitutes to high to be useful
        parent = self            # NOTE: using 'None' for parents does not work with surrogate model
        self.suppress_output = True   # Suppress the console output of particle swarm
        self.allow_update = True      # Allow objective call to update state 
        
        useSurrogateModel  = True
        self.sm_approx = None       # the approximator 
        self.sm_opt = None          # the surrogate model optimizer
        self.sm_opt_df = None       # dataframe with surrogate model optimizer params
        self.b_opt  = None          # the base  optimizer
        self.opt_df = None          # dataframe with optimizer params
        

        # SURROGATE MODEL OPTIMIZER SETUP (2nd optimizer first)
        # 1) set up df for later use
        # 2) pass in class. DO NOT set here.
        if SURROGATE_OPT_CHOICE == 0:
            # 0: pso_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'T_MOD': [0.65],              # Variable time-step extinction coefficient
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'VLIM':  [0.5] }              # Initial velocity limit

            self.sm_opt_df  = pd.DataFrame(opt_params)
            self.sm_opt  = pso_p_swarm
            
        elif SURROGATE_OPT_CHOICE == 1:
            # 1: pso_basic
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'VLIM':  [0.5] }              # Initial velocity limit

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = pso_b_swarm  
                            
        elif SURROGATE_OPT_CHOICE == 2:
            # 2: pso_quantum
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'BETA':  [0.5] }              #Float constant controlling influence 
                                                      #       between the personal and global best positions

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = pso_q_swarm 

        elif SURROGATE_OPT_CHOICE == 3:
            # 3: cat_swarm_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [8],   # Number of particles in swarm
                        'BOUNDARY': [1],            # int boundary 1 = random,      2 = reflecting
                                                    #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[2]],           # Update vector weights
                        'VLIM':  [1.5],             # Initial velocity limit
                        'MR': [0.02],               # Mixture Ratio (MR). Small value for tracing population %.
                        'SMP': [5],                 # Seeking memory pool. Num copies of cats made.
                        'SRD': [0.45],              # Seeking range of the selected dimension. 
                        'CDC': [2],                 # Counts of dimension to change. mutation.
                        'SPC': True}              # self-position consideration. boolean.

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = catswarm    

        elif SURROGATE_OPT_CHOICE == 4:
            #4: sand_cat_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [8],     # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[2, 2.2, 2]]]}   # Update vector weights

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = sand_cat_swarm 


        elif SURROGATE_OPT_CHOICE == 5:
            # 5: cat_swarm_quantum
            # Constant variables
            oopt_params = {'NO_OF_PARTICLES': [8],   # Number of particles in swarm
                        'BOUNDARY': [1],            # int boundary 1 = random,      2 = reflecting
                                                    #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[2]],           # Update vector weights
                        'MR': [0.02],               # Mixture Ratio (MR). Small value for tracing population %.
                        'SMP': [5],                 # Seeking memory pool. Num copies of cats made.
                        'SRD': [0.45],              # Seeking range of the selected dimension. 
                        'CDC': [2],                 # Counts of dimension to change. mutation.
                        'SPC': True,                # self-position consideration. boolean.
                        'BETA': [0.5] }             #Float constant controlling influence 
                                                    #     between the personal and global best positions
            
            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = cat_q_swarm 

        elif SURROGATE_OPT_CHOICE == 6:
            # 6: chicken_swarm_python
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70]}              # Reorganize groups every G steps 

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = chicken_swarm   
        
        elif SURROGATE_OPT_CHOICE == 7:
            # 7: 2015_improved_chicken_swarm
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70],              # Reorganize groups every G steps 
                        'MIN_WEIGHT': [0.4], 
                        'MAX_WEIGHT': [0.9],
                        'LEARNING_CONSTANT': [0.4]}

            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = chicken_i15_swarm 


        elif SURROGATE_OPT_CHOICE == 8:
            # 8: chicken_swarm_quantum
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70],              # Reorganize groups every G steps 
                        'BETA': [0.5],          # Float constant controlling influence 
                                              #     between the personal and global best positions 
                        'QUANTUM_ROOSTERS': True} # Boolean. Use quantum rooster or classical movement
            
            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = chicken_q_swarm 
        

        elif SURROGATE_OPT_CHOICE == 9:
            # 9: multi_glods_python
            # Constant variables
            opt_params = {'BP': [0.5],               # Beta Par
                        'GP': [1],                   # Gamma Par
                        'SF': [1] }                  # Search Frequency
            
            self.sm_opt_df = pd.DataFrame(opt_params)
            self.sm_opt  = multi_glods 

        else:
            print("unknown surrogate model selected with option: " + str(BASE_OPT_CHOICE))
            return



        # SURROGATE MODEL APPROXIMATOR SETUP
        # estimates the objective function for the surrogate model
        if APPROXIMATOR_CHOICE == 0:
            # RBF Network vars
            RBF_kernel  = 'gaussian' #options: 'gaussian', 'multiquadric'
            RBF_epsilon = 1.0
            num_init_points = 1
            self.sm_approx = RBFNetwork(kernel=RBF_kernel, epsilon=RBF_epsilon)  
            noError, errMsg = self.sm_approx._check_configuration(num_init_points, RBF_kernel)

        elif APPROXIMATOR_CHOICE == 1:
            # Gaussian Process vars
            GP_noise = 1e-10
            GP_length_scale = 1.0
            num_init_points = 3
            self.sm_approx = GaussianProcess(length_scale=GP_length_scale,noise=GP_noise) 
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 2:
            # Kriging vars
            K_noise = 1e-10
            K_length_scale = 1.0   
            num_init_points = 2 
            self.sm_approx = Kriging(length_scale=K_length_scale, noise=K_noise)
            
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 3:
            # Polynomial Regression vars
            PR_degree = 5
            num_init_points = 1
            self.sm_approx = PolynomialRegression(degree=PR_degree)
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 4:
            # Polynomial Chaos Expansion vars
            PC_degree = 5 
            num_init_points = 1
            self.sm_approx = PolynomialChaosExpansion(degree=PC_degree)
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 5:
            # KNN regression vars
            KNN_n_neighbors=3
            KNN_weights='uniform'  #options: 'uniform', 'distance'
            num_init_points = 1
            self.sm_approx = KNNRegression(n_neighbors=KNN_n_neighbors, weights=KNN_weights)
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 6:
            # Decision Tree Regression vars
            DTR_max_depth = 5  # options: ints
            num_init_points = 1
            self.sm_approx = DecisionTreeRegression(max_depth=DTR_max_depth)
            noError, errMsg = self.sm_approx._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 7:
            # Matern Process vars
            DTR_max_depth = 1  # options: ints
            num_init_points = 1
            MP_length_scale = 1.1
            MP_noise = 1e-10
            MP_nu = 3/2
            self.sm = MaternProcess(length_scale=MP_length_scale, noise=MP_noise, nu=MP_nu)
            noError, errMsg = self.sm._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 8:
            # Lagrangian penalty linear regression vars
            num_init_points = 2
            LLReg_noise = 1e-10
            LLReg_constraint_degree=1
            self.sm = LagrangianLinearRegression(noise=LLReg_noise, constraint_degree=LLReg_constraint_degree)
            noError, errMsg = self.sm._check_configuration(num_init_points)

        elif APPROXIMATOR_CHOICE == 9:
            # Lagrangian penalty polynomial regression vars
            num_init_points = 2
            LPReg_degree = 5
            LPReg_noise = 1e-10
            LPReg_constraint_degree = 3
            self.sm = LagrangianPolynomialRegression(degree=LPReg_degree, noise=LPReg_noise, constraint_degree=LPReg_constraint_degree)
            noError, errMsg = self.sm._check_configuration(num_init_points)


        if noError == False:
            print("ERROR in development_test.py. Incorrect approximator model configuration")
            print(errMsg)
            return



        # BASE OPTIMIZER SETUP
        if BASE_OPT_CHOICE == 0:
            # 0: pso_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'T_MOD': [0.65],              # Variable time-step extinction coefficient
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                    #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'VLIM':  [0.5] }              # Initial velocity limit

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = pso_p_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)  
            
        elif BASE_OPT_CHOICE == 1:
            # 1: pso_basic
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'VLIM':  [0.5] }              # Initial velocity limit


            opt_df = pd.DataFrame(opt_params)
            self.b_opt = pso_b_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)  
                            
        elif BASE_OPT_CHOICE == 2:
            # 2: pso_quantum
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [50],    # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[0.7, 1.5, 0.5]]], # Update vector weights
                        'BETA':  [0.5] }              #Float constant controlling influence 
                                                      #       between the personal and global best positions

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = pso_q_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent,  
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)    

        elif BASE_OPT_CHOICE == 3:
            # 3: cat_swarm_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [8],   # Number of particles in swarm
                        'BOUNDARY': [1],            # int boundary 1 = random,      2 = reflecting
                                                    #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[2]],           # Update vector weights
                        'VLIM':  [1.5],             # Initial velocity limit
                        'MR': [0.02],               # Mixture Ratio (MR). Small value for tracing population %.
                        'SMP': [5],                 # Seeking memory pool. Num copies of cats made.
                        'SRD': [0.45],              # Seeking range of the selected dimension. 
                        'CDC': [2],                 # Counts of dimension to change. mutation.
                        'SPC': True}              # self-position consideration. boolean.

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = catswarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)     

        elif BASE_OPT_CHOICE == 4:
            # 4: sand_cat_python
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [8],     # Number of particles in swarm
                        'BOUNDARY': [1],              # int boundary 1 = random,      2 = reflecting
                                                      #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[[2, 2.2, 2]]]}   # Update vector weights

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = sand_cat_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)   
            
        elif BASE_OPT_CHOICE == 5:
            # 5: cat_swarm_quantum
            # Constant variables
            opt_params = {'NO_OF_PARTICLES': [8],   # Number of particles in swarm
                        'BOUNDARY': [1],            # int boundary 1 = random,      2 = reflecting
                                                    #              3 = absorbing,   4 = invisible
                        'WEIGHTS': [[2]],           # Update vector weights
                        'MR': [0.02],               # Mixture Ratio (MR). Small value for tracing population %.
                        'SMP': [5],                 # Seeking memory pool. Num copies of cats made.
                        'SRD': [0.45],              # Seeking range of the selected dimension. 
                        'CDC': [2],                 # Counts of dimension to change. mutation.
                        'SPC': True,                # self-position consideration. boolean.
                        'BETA': [0.5] }             #Float constant controlling influence 
                                                    #     between the personal and global best positions
            
            opt_df = pd.DataFrame(opt_params)
            self.b_opt = cat_q_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)     

        elif BASE_OPT_CHOICE == 6:
            # 6: chicken_swarm_python
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70]}              # Reorganize groups every G steps 

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = chicken_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)    
        
        elif BASE_OPT_CHOICE == 7:
            # 7: 2015_improved_chicken_swarm
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70],              # Reorganize groups every G steps 
                        'MIN_WEIGHT': [0.4], 
                        'MAX_WEIGHT': [0.9],
                        'LEARNING_CONSTANT': [0.4]}

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = chicken_i15_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)    
        
        elif BASE_OPT_CHOICE == 8:
            # 8: chicken_swarm_quantum
            # Constant variables
            opt_params = {'BOUNDARY': [1],      # int boundary 1 = random,      2 = reflecting
                                                #              3 = absorbing,   4 = invisible
                        'RN': [10],             # Total number of roosters
                        'HN': [20],             # Total number of hens
                        'MN': [15],             # Number of mother hens in total hens
                        'CN': [20],             # Total number of chicks
                        'G': [70],              # Reorganize groups every G steps 
                        'BETA': [0.5],          # Float constant controlling influence 
                                              #     between the personal and global best positions 
                        'QUANTUM_ROOSTERS': True} # Boolean. Use quantum rooster or classical movement
            
            opt_df = pd.DataFrame(opt_params)
            self.b_opt = chicken_q_swarm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)    
        

        elif BASE_OPT_CHOICE == 9:
            # 9: multi_glods_python
            # Constant variables
            opt_params = {'BP': [0.5],               # Beta Par
                        'GP': [1],                   # Gamma Par
                        'SF': [1] }                  # Search Frequency

            opt_df = pd.DataFrame(opt_params)
            self.b_opt = multi_glods(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt)   
        else:
            print("unknown surrogate model selected with option: " + str(BASE_OPT_CHOICE))
            return



    # SURROGATE MODEL FUNCS
    # create the 'surrogate objective function' for the surrogate model optimizer to optimize   
    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.sm_approx.fit(x,y)
        

    def model_predict(self, x, output_size=None):
        # call out to parent class to use surrogate model
        #'mean' is regressive definition. not statistical
        #'variance' only applies for some surrogate models
        if output_size == None:
            output_size = self.out_vars

        mean, noErrors = self.sm_approx.predict(x, output_size)
        return mean, noErrors

    def model_get_variance(self):
        variance = self.sm_approx.calculate_variance()
        return variance


    def fit_and_create_surogate(self, opt_M, opt_F_Pb, surrogateOptimizer):
        # opt_M : 'base' optimizer particle locs
        # opt_F_Pb: 'base' optimizer personal best fitness
        # surrogateOptimizer : the surrogate optimizer class obj. 
    
        #called at the controller level so that the params don't 
        # get passed down and then used at this level anyways

        # fit surrogate model using current particle positions
        # this model needs to be fit to create something that can then be modeled
        self.fit_model(opt_M, opt_F_Pb)

        # define objective function pass through via parent 
        func_f = self.model_predict   


        # to make models modular & not deal with 
        # re-converting values or storing duplicates, the surrogate optimizer
        # is set up here. 

        setup_sm_opt = surrogateOptimizer(self.LB, self.UB, self.TARGETS, self.sm_tol, self.sm_maxit,  
                                                         obj_func=func_f, constr_func=self.constr_F, 
                                                         opt_df=self.sm_opt_df,
                                                         parent=self)

        return setup_sm_opt

    def get_surrogate_model_settings(self):
        # returns the parameters for creating the surrogate model optimizer
        # the surrogate model optimizer is created new every time it is called
        return self.sm_opt_df, self.sm_tol, self.sm_maxit


    # DEBUG
    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)

    def record_params(self):
        # this function is called from particle_swarm.py to trigger a write to a log file
        # running in the AntennaCAT GUI to record the parameter iteration that caused an error
        pass
         
    
    # RUN
    def run(self):
    
        # instantiation of particle swarm optimizer 
        while not self.b_opt.complete():

            # step through optimizer processing
            # update_velocity, will change the particle location
            self.b_opt.step(self.suppress_output)

            # call the objective function, control 
            # when it is allowed to update and return 
            # control to optimizer

            # for some objective functions, the function
            # might not evaluate correctly (e.g., under/overflow)
            # so when that happens, the function is not evaluated
            # and the 'step' fucntion will re-gen values and try again

            self.b_opt.call_objective(self.allow_update)
            iter, eval = self.b_opt.get_convergence_data()
            if (eval < self.best_eval) and (eval != 0):
                self.best_eval = eval
            if self.suppress_output:
                if iter%100 ==0: #print out every 100th iteration update
                    print("*************************************************")
                    print("BASE OPTIMIZER Iteration")
                    print(iter)
                    print("BASE OPTIMIZER Best Eval")
                    print(self.best_eval)
        
        print("*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~")
        print("OPTIMIZATION COMPLETE:")
        print("BASE OPTIMIZER Iteration")
        print(iter)
        print("BASE OPTIMIZER Best Eval")
        print(self.best_eval)
        print("Optimized Solution")
        print(self.b_opt.get_optimized_soln())
        print("Optimized Outputs")
        print(self.b_opt.get_optimized_outs())





if __name__ == "__main__":

    mt = MainTest()
    mt.run()