# 24/03/2021, B. M. Giblin, Postdoc, Edinburgh

# General code to run an MCMC likelihood analysis executing the emulator at each step in the chain.
# This accepts a parameter file which specifies the training set, hyperparameters, 
# data vector & covariance for each statistic and details on which statistics to combine.
# Also contains MCMC details such as number of steps, truth values (if any), priors etc,
# plus the name of a directory where all results should be saved.
# Returns the samples for each MCMC, makes a nice plot, and returns the name these are saved under.

import numpy as np
import sys
from Classes_4_Lhd import Get_Input 

Run_MCMC = True

paramfile = sys.argv[1]
GI = Get_Input(paramfile)   

if Run_MCMC:
	# RUN THE MCMC ANEW....
	Samples_Stats, Savename_Stats, Samples_Comb, Savename_Comb = GI.Run_Analysis_MCMC()


else:
	# ....OR.....
	# ...LOAD THE PRE-SAVED SAMPLES AND PLOT THEM
	savename_plot = None            # setting savename of plot to None means it will use default
	Constraints_Comb = GI.Plot_MCMC_Lhd_Multiple(None)    






