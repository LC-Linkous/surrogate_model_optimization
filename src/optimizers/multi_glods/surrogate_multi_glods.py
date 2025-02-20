#! /usr/bin/python3

##--------------------------------------------------------------------\
#   surrogate_model_optimization
#   './surrogate_model_optimization/src/optimizers/multi_glods/multi_glods.py'
#   Class for intializing and interfacing with the multiGLODS algorithm
#   NOTE: multiglods.py is the statemachine, 
#       and multiglods_ctl.py is the controller
#
#
#
#   Author(s): Jonathan Lundquist, Lauren Linkous 
#   Last update: February 18, 2024
##--------------------------------------------------------------------\


import numpy as np
import sys

try: # for outside func calls, program calls
    sys.path.insert(0, './surrogate_model_optimization/src/optimizers/multi_glods/')
    from multiglods_ctl import one_time_init
    from multiglods_helpers import f_eval_objective_call
    from multiglods import multiglods

except:# for local, unit testing
    from optimizers.multi_glods.multiglods_ctl import one_time_init
    from optimizers.multi_glods.multiglods_helpers import f_eval_objective_call
    from optimizers.multi_glods.multiglods import multiglods

class multi_glods:
    # arguments should take form: 
    # multi_glods([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj, bool,
    # bool, class obj) 
    #  
    # opt_df contains class-specific tuning parameters
    # BP: float
    # GP: int
    # SF: int
    #

    def __init__(self,  LB, UB, TARGETS, TOL, MAXIT,
                    obj_func, constr_func, 
                    opt_df,
                    parent=None, detailedWarnings=False, 
                    useSurrogateModel=False, surrogateOptimizer=None): 

        # defaults - DO NOT USE SURROGATE INSIDE MULTIGLODS
        # this compatibility does not exist yet. 
        useSurrogateModel=False
        secondOptimizer=None

        #reformat since multi_glods needs single []
        LB = LB[0]
        UB = UB[0]
        TARGETS = TARGETS

        #unpack the opt_df standardized vals
        NO_OF_VARS = len(LB)
        BP = opt_df['BP'][0]
        GP = opt_df['GP'][0]
        SF = opt_df['SF'][0]
   
        self.init, self.run_ctl, self.alg, \
            self.prob, self.ctl, self.state = \
                one_time_init(NO_OF_VARS, LB, UB, TARGETS, TOL, MAXIT,
                              BP, GP, SF, obj_func, constr_func)


        self.prob['parent'] = parent
        self.detailedWarnings = detailedWarnings
        self.done = 0

    def step(self, suppress_output):
        self.done, self.init, self.run_ctl, \
            self.alg, self.prob, self.ctl, self.state = \
                multiglods(self.init, self.run_ctl, self.alg, 
                           self.prob, self.ctl, self.state, 
                           suppress_output)
        
   
    def call_objective(self, allow_update):
        self.state, self.prob = f_eval_objective_call(self.state, 
                                                      self.prob, 
                                                      self.ctl,
                                                      allow_update)
        
    def export_glods(self):
        glods_export = {'init': self.init, 'run_ctl': self.run_ctl,
                        'alg': self.alg, 'prob': self.prob,
                        'ctl': self.ctl, 'state': self.state}
        return glods_export
    
    def import_glods(self, glods_export, obj_func):
        self.init = glods_export['init']
        self.run_ctl = glods_export['run_ctl']
        self.alg = glods_export['alg']
        self.prob = glods_export['prob']
        self.ctl = glods_export['ctl']
        self.state = glods_export['state']
        self.ctl['obj_func'] = obj_func

    def complete(self):
        return self.done
    
    def get_obj_inputs(self):
        if self.state['init']:
            return self.init['x_ini']
        else:
            return self.prob['xtemp']
        
    def get_convergence_data(self):
        if len(np.shape(self.ctl['Flist'])) > 1:
            best_eval = np.linalg.norm(self.ctl['Flist'][:,0])
        else:
            best_eval = np.linalg.norm(self.ctl['Flist'])

        # return iteration for objective function call.
        #  There's several 'iter' counters.
        #  self.run_ctl['iter'] : 
        #  self.ctl['objective_iter'] : objective func call counter   
    
        iteration = 1*self.ctl['objective_iter']
        return iteration, best_eval 

    def get_optimized_soln(self):
        soln = np.vstack(self.prob['Plist'][:,0])
        return soln
    
    def get_optimized_outs(self):
        soln = np.vstack(self.ctl['Flist'][:,0])
        return soln
    

    # for plotting
    def get_search_locations(self):
        x_locations = self.prob['Plist'] 
        return x_locations

    def get_fitness_values(self):
        x_locations = self.ctl['Flist']
        return x_locations