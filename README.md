# surrogate_model_optimization

This repository features the core AntennaCAT optimizer set with surrogate model capabilities. It is specifically for experimental work as features are added for the main [AntennaCAT software](https://github.com/LC-Linkous/AntennaCalculationAutotuningTool) and added as a level of transparency. This repo is, when stable, for unit testing and development. This should NOT be the entry point for testing as it has fewer debug options.  

For the specifics of each optimizer, refer to their repository or branch. The original versions of the optimizers can be found at:

| Base Optimizer | Alternate Version | Quantum-Inspired Optimizer | Surrogate Model Version |
| ------------- | ------------- | ------------- |------------- |
| [pso_python](https://github.com/LC-Linkous/pso_python) | [pso_basic](https://github.com/LC-Linkous/pso_python/tree/pso_basic) | [pso_quantum](https://github.com/LC-Linkous/pso_quantum)  | all versions are options in [surrogate_model_optimization](https://github.com/LC-Linkous/surrogate_model_optimization)|
| [cat_swarm_python](https://github.com/LC-Linkous/cat_swarm_python) | [sand_cat_python](https://github.com/LC-Linkous/cat_swarm_python/tree/sand_cat_python)| [cat_swarm_quantum](https://github.com/LC-Linkous/cat_swarm_python/tree/cat_swarm_quantum) |all versions are options in [surrogate_model_optimization](https://github.com/LC-Linkous/surrogate_model_optimization) |
| [chicken_swarm_python](https://github.com/LC-Linkous/chicken_swarm_python) | [2015_improved_chicken_swarm](https://github.com/LC-Linkous/chicken_swarm_python/tree/improved_chicken_swarm) <br>2022 improved chicken swarm| [chicken_swarm_quantum](https://github.com/LC-Linkous/chicken_swarm_python/tree/chicken_swarm_quantum)  | all versions are options in [surrogate_model_optimization](https://github.com/LC-Linkous/surrogate_model_optimization)|
| [sweep_python](https://github.com/LC-Linkous/sweep_python)  | *alternates in base repo | -  | - |
| [bayesian optimization_python](https://github.com/LC-Linkous/bayesian_optimization_python)  | -| - | *interchangeable surrogate models <br> included in base repo |
| [multi_glods_python](https://github.com/LC-Linkous/multi_glods_python)| GLODS <br> DIRECT | - | multiGLODS option in [surrogate_model_optimization](https://github.com/LC-Linkous/surrogate_model_optimization)|



The surrogate model approximators were originally featured in [bayesian_optimization_python](https://github.com/LC-Linkous/bayesian_optimization_python) by [LC-Linkous](https://github.com/LC-Linkous).  See [References](#references) for the running list of references as optimizers and surrogate models approximators are added/edited, and features are updated. The references are in no particular order and are intended as a collection of relevant information for anyone interested in more information. Actual citations for optimizers or a process can be found in the original optimizer repos or the associated publications.

## Table of Contents
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Constraint Handling](#constraint-handling)
    * [Internal Objective Function Examples](#internal-objective-function-examples)
* [Example Testing](#example-implementations)
* [References](#references)
* [Publications and Integration](#publications-and-integration)

## Requirements

This project requires numpy and pandas for the optimization models. matplotlib is used for creating a preview of the mathematical functions used with the objective function calls.

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.3.2
cycler==0.12.1
fonttools==4.58.4
kiwisolver==1.4.8
matplotlib==3.10.3
numpy==2.2.3
packaging==25.0
pandas==2.2.3
pillow==11.2.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.1
six==1.17.0
tzdata==2025.1

```

For manual installation:

```python
pip install numpy, pandas, matplotlib

```


## Implementation

### Initialization 

Initialization is currently being streamlined. It is dependent based on the layering of the surrogate models and base optimizer. There is no elegant solution for development, but it is well documented. 

The `./src/development_test.py` file is the main entry point for this repo. This file contains:

1. Setup and configuration for the approximation functions used to create the surrogate models
2. Setup and configuration for the surrogate model optimizer, sometimes referred to as the 'internal optimizer'
3. Setup and configuration for the driving optimizer, sometimes referred to as the 'base optimizer'.

A the top of the file, there are three `CONSTANTS` used to set integer values for an if/else statement. All optimizers and approximator functions are set with defaults.

```python
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
        SURROGATE_OPT_CHOICE = 1
        APPROXIMATOR_CHOICE = 2

```

Default class values are set next.

```python

        # OPTIMIZER INIT
        self.best_eval = 10      #usually set to 1 because anything higher is too many magnitudes to high to be useful
        parent = self            # NOTE: using 'None' for parents does not work with surrogate model
        self.evaluate_threshold = False # use target or threshold. True = THRESHOLD, False = EXACT TARGET
        self.suppress_output = True   # Suppress the console output of particle swarm
        self.allow_update = True      # Allow objective call to update state 
        
        useSurrogateModel  = True   # The optimizers can be run without an internal optimizer
        self.sm_approx = None       # the approximator 
        self.sm_opt = None          # the surrogate model optimizer
        self.sm_opt_df = None       # dataframe with surrogate model optimizer params
        self.b_opt  = None          # the base  optimizer
        self.opt_df = None          # dataframe with optimizer params
        
```

When the surrogate optimizers are set, the CLASSES and DATAFRAME are saved as variables. The driving optimizer will create the internal optimizer as needed using the same LIMITS, BOUNDARY, TARGETS, and THESHOLDS shared by both. 

```python
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

    [...]
```            

The approximation functions are configured and initialized once, and then used for the optimization. These functions have memory.

```python
        if APPROXIMATOR_CHOICE == 0:
            # RBF Network vars
            RBF_kernel  = 'gaussian' #options: 'gaussian', 'multiquadric'
            RBF_epsilon = 1.0
            num_init_points = 1
            self.sm_approx = RBFNetwork(kernel=RBF_kernel, epsilon=RBF_epsilon)  
            noError, errMsg = self.sm_approx._check_configuration(num_init_points, RBF_kernel)

    [...]
```            

The driving optimizer, or base optimizer, is set last so that the internal optimizer can be passed in. 

```python
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
                                    evaluate_threshold=self.evaluate_threshold, 
                                    obj_threshold=self.THRESHOLD, 
                                    useSurrogateModel=useSurrogateModel, 
                                    surrogateOptimizer=self.sm_opt,
                                    decimal_limit=4)  
            

    [...]
```            



### Constraint Handling
Users must create their own constraint function for their problems, if there are constraints beyond the problem bounds.  This is then passed into the constructor. If the default constraint function is used, it always returns true (which means there are no constraints).

### Boundary Types

Most optimizers have 4 different types of bounds, Random (Particles that leave the area respawn), Reflection (Particles that hit the bounds reflect), Absorb (Particles that hit the bounds lose velocity in that direction), Invisible (Out of bound particles are no longer evaluated).

Some updates have not incorporated appropriate handling for all boundary conditions. This bug is known and is being worked on. The most consistent boundary type at the moment is Random. If constraints are violated, but bounds are not, currently random bound rules are used to deal with this problem. 


### Multi-Objective Optimization
The no preference method of multi-objective optimization, but a Pareto Front is not calculated. Instead, the best choice (smallest norm of output vectors) is listed as the output.

### Objective Function Handling

The objective function is handled in two parts. 

* First, a defined function, such as one passed in from `func_F.py` (see examples), is evaluated based on current particle locations. This allows for the optimizers to be utilized in the context of 1. benchmark functions from the objective function library, 2. user defined functions, 3. replacing explicitly defined functions with outside calls to programs such as simulations or other scripts that return a matrix of evaluated outputs. 

* Secondly, the actual objective function is evaluated. In the AntennaCAT set of optimizers, the objective function evaluation is either a `TARGET` or `THRESHOLD` evaluation. For a `TARGET` evaluation, which is the default behavior, the optimizer minimizes the absolute value of the difference of the target outputs and the evaluated outputs. A `THRESHOLD` evaluation includes boolean logic to determine if a 'greater than or equal to' or 'less than or equal to' or 'equal to' relation between the target outputs (or thresholds) and the evaluated outputs exist. 

Future versions may include options for function minimization when target values are absent. 


#### Creating a Custom Objective Function

Custom objective functions can be used by creating a directory with the following files:
* configs_F.py
* constr_F.py
* func_F.py

`configs_F.py` contains lower bounds, upper bounds, the number of input variables, the number of output variables, the target values, and a global minimum if known. This file is used primarily for unit testing and evaluation of accuracy. If these values are not known, or are dynamic, then they can be included experimentally in the controller that runs the optimizer's state machine. 

`constr_F.py` contains a function called `constr_F` that takes in an array, `X`, of particle positions to determine if the particle or agent is in a valid or invalid location. 

`func_F.py` contains the objective function, `func_F`, which takes two inputs. The first input, `X`, is the array of particle or agent positions. The second input, `NO_OF_OUTS`, is the integer number of output variables, which is used to set the array size. In included objective functions, the default value is hardcoded to work with the specific objective function.

Below are examples of the format for these files.

`configs_F.py`:
```python
OBJECTIVE_FUNC = func_F
CONSTR_FUNC = constr_F
OBJECTIVE_FUNC_NAME = "one_dim_x_test.func_F" #format: FUNCTION NAME.FUNCTION
CONSTR_FUNC_NAME = "one_dim_x_test.constr_F" #format: FUNCTION NAME.FUNCTION

# problem dependent variables
LB = [[0]]             # Lower boundaries
UB = [[1]]             # Upper boundaries
IN_VARS = 1            # Number of input variables (x-values)
OUT_VARS = 1           # Number of output variables (y-values) 
TARGETS = [0]          # Target values for output
GLOBAL_MIN = []        # Global minima sample, if they exist. 

```

`constr_F.py`, with no constraints:
```python
def constr_F(x):
    F = True
    return F
```

`constr_F.py`, with constraints:
```python
def constr_F(X):
    F = True
    # objective function/problem constraints
    if (X[2] > X[0]/2) or (X[2] < 0.1):
        F = False
    return F
```

`func_F.py`:
```python
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
        noErrors = False

    return [F], noErrors
```

#### Internal Objective Function Example

There are three functions included in the repository:
1) Himmelblau's function, which takes 2 inputs and has 1 output
2) A multi-objective function with 3 inputs and 2 outputs (see lundquist_3_var)
3) A single-objective function with 1 input and 1 output (see one_dim_x_test)

Each function has four files in a directory:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.
   4) graph.py - contains a script to graph the function for visualization.

Other multi-objective functions can be applied to this project by following the same format (and several have been collected into a compatible library, and will be released in a separate repo)

<p align="center">
        <img src="media/himmelblau_plots.png" alt="Himmelblau’s function" height="250">
</p>
   <p align="center">Plotted Himmelblau’s Function with 3D Plot on the Left, and a 2D Contour on the Right</p>

```math
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

| Global Minima | Boundary | Constraints |
|----------|----------|----------|
| f(3, 2) = 0                 | $-5 \leq x,y \leq 5$  |   | 
| f(-2.805118, 3.121212) = 0  | $-5 \leq x,y \leq 5$  |   | 
| f(-3.779310, -3.283186) = 0 | $-5 \leq x,y \leq 5$  |   | 
| f(3.584428, -1.848126) = 0  | $-5 \leq x,y \leq 5$   |   | 

<p align="center">
        <img src="media/obj_func_pareto.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Multi-Objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 

<p align="center">
        <img src="media/1D_test_plots.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Single Input, Single-objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
f(\mathbf{x}) = sin(5 * x^3) + cos(5 * x) * (1 - tanh(x^2))
```
| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 1      | $0\leq x\leq 1$  | $0\leq x\leq 1$| |

Local minima at $(0.444453, -0.0630916)$

Global minima at $(0.974857, -0.954872)$

### Target vs. Threshold Configuration

An April 2025 feature is the user ability to toggle TARGET and THRESHOLD evaluation for the optimized values. The key variables for this are:

```python
# Boolean. use target or threshold. True = THRESHOLD, False = EXACT TARGET
evaluate_threshold = True  

# array
TARGETS = func_configs.TARGETS    # Target values for output from function configs
# OR:
TARGETS = [0,0,0] #manually set BASED ON PROBLEM DIMENSIONS

# threshold is same dims as TARGETS
# 0 = use target value as actual target. value should EQUAL target
# 1 = use as threshold. value should be LESS THAN OR EQUAL to target
# 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
#DEFAULT THRESHOLD
THRESHOLD = np.zeros_like(TARGETS) 
# OR
THRESHOLD = [0,1,2] # can be any mix of TARGET and THRESHOLD  
```

To implement this, the original `self.Flist` objective function calculation has been replaced with the function `objective_function_evaluation`, which returns a numpy array.

The original calculation:
```python
self.Flist = abs(self.targets - self.Fvals)
```
Where `self.Fvals` is a re-arranged and error checked returned value from the passed in function from `func_F.py` (see examples for the internal objective function or creating a custom objective function). 

When using a THRESHOLD, the `Flist` value corresponding to the target is set to epsilon (the smallest system value) if the evaluated `func_F` value meets the threshold condition for that target item. If the threshold is not met, the absolute value of the difference of the target output and the evaluated output is used. With a THRESHOLD configuration, each value in the numpy array is evaluated individually, so some values can be 'greater than or equal to' the target while others are 'equal' or 'less than or equal to' the target. 


## Example Testing

Currently, `development_test.py` is the only file for testing multiple combinations of base optimizers, surrogate models, and linear approximators. 

To choose one of the 3 included objective functions:
The internal objective functions are included at the top of the file. Uncomment ONE of the functions to use it. If multiple imports are uncommented, the lowest function is the one the compiler will use.

```python
# OBJECTIVE FUNCTION SELECTION -UNCOMMENT HERE TO SELECT
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs     # multi objective function
```

To select optimizers and approximators:

```python
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
        SURROGATE_OPT_CHOICE = 1
        APPROXIMATOR_CHOICE = 2

```

Individual optimizer parameters currently need to be adjusted with optimizer creation. In the future, this will be more streamlined for unit testing.

To change the default values of the optimizers or approximation functions, they need to be manually edited within the `IF/ELSE` structure. There are too many variables to have an collection at the top. 



## References

[1] J. Kennedy and R. Eberhart, "Particle swarm optimization," Proceedings of ICNN'95 - International Conference on Neural Networks, Perth, WA, Australia, 1995, pp. 1942-1948 vol.4, doi: 10.1109/ICNN.1995.488968.

[2] A. L. Custódio and J. F. A. Madeira, “MultiGLODS: global and local multiobjective optimization using direct search,” Journal of Global Optimization, vol. 72, no. 2, pp. 323–345, Feb. 2018, doi: https://doi.org/10.1007/s10898-018-0618-1.

[3] A. L. Custódio and J. F. A. Madeira, “GLODS: Global and Local Optimization using Direct Search,” Journal of Global Optimization, vol. 62, no. 1, pp. 1–28, Aug. 2014, doi: https://doi.org/10.1007/s10898-014-0224-9.

[4] Wikipedia Contributors, “Himmelblau’s function,” Wikipedia, Dec. 29, 2023. https://en.wikipedia.org/wiki/Himmelblau%27s_function

[5] Wikipedia Contributors, “Bayesian optimization,” Wikipedia, Jul. 05, 2019. https://en.wikipedia.org/wiki/Bayesian_optimization

[6] W. Wang, “Bayesian Optimization Concept Explained in Layman Terms,” Medium, Mar. 22, 2022. https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f

[7] C. Brecque, “The intuitions behind Bayesian Optimization with Gaussian Processes,” Medium, Apr. 02, 2021. https://towardsdatascience.com/the-intuitions-behind-bayesian-optimization-with-gaussian-processes-7e00fcc898a0

[8] “Introduction to Bayesian Optimization (BO) — limbo 0.1 documentation,” resibots.eu. https://resibots.eu/limbo/guides/bo.html

[9] “Radial Basis Function Networks (RBFNs) with Python 3: A Comprehensive Guide – Innovate Yourself,” Nov. 03, 2023. https://innovationyourself.com/radial-basis-function-networks-rbfn/

[10] Everton Gomede, PhD, “Radial Basis Functions Neural Networks: Unlocking the Power of Nonlinearity,” Medium, Jun. 06, 2023. https://medium.com/@evertongomede/radial-basis-functions-neural-networks-unlocking-the-power-of-nonlinearity-c67f6240a5bb

[11] J. Luo, W. Xu and J. Chen, "A Novel Radial Basis Function (RBF) Network for Bayesian Optimization," 2021 IEEE 7th International Conference on Cloud Computing and Intelligent Systems (CCIS), Xi'an, China, 2021, pp. 250-254, doi: 10.1109/CCIS53392.2021.9754629.

[12] Wikipedia Contributors, “Kriging,” Wikipedia, Oct. 16, 2018. https://en.wikipedia.org/wiki/Kriging

[13] “Polynomial kernel,” Wikipedia, Oct. 02, 2019. https://en.wikipedia.org/wiki/Polynomial_kernel

[14] A. Radhakrishnan, M. Luyten, G. Stefanakis, and C. Cai, “Lecture 3: Kernel Regression,” 2022. Available: https://web.mit.edu/modernml/course/lectures/MLClassLecture3.pdf

[15] “Polynomial chaos,” Wikipedia, Mar. 19, 2024. https://en.wikipedia.org/wiki/Polynomial_chaos

[16] “Polynomial Chaos Expansion — Uncertainty Quantification,” dictionary.helmholtz-uq.de. https://dictionary.helmholtz-uq.de/content/PCE.html (accessed Jun. 28, 2024).

[17] T. Srivastava, “Introduction to KNN, K-Nearest Neighbors : Simplified,” Analytics Vidhya, Mar. 07, 2019. https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/

[18] Wikipedia Contributors, “k-nearest neighbors algorithm,” Wikipedia, Mar. 19, 2019. https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

[19] “Python | Decision Tree Regression using sklearn,” GeeksforGeeks, Oct. 04, 2018. https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/

[20] “Decision Tree Regression,” Saedsayad.com, 2019. https://www.saedsayad.com/decision_tree_reg.htm

[21] Wikipedia Contributors, “Decision tree learning,” Wikipedia, Jun. 12, 2019. https://en.wikipedia.org/wiki/Decision_tree_learning

## Publications and Integration
This software works as a stand-alone implementation, and as a selection of optimizers integrated into AntennaCAT. Publications featuring the code as part of AntennaCAT will be added as they become public.


