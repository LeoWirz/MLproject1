from enum import Enum

class Model(Enum):
    LEAST_SQUARES_GD 		= 0
    LEAST_SQUARES_SGD       = 1
    LEAST_SQUARES           = 2
    RIGDE_REGRESSION    	= 3
    LOGISTIC_REGRESSION     = 4
    REG_LOGISTIC_REGRESSION = 5

def joke():
	print("joke on you")
