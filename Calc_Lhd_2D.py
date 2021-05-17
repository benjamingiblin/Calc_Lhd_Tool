# 02/12/2020, B. Giblin, Edinburgh
# Re-write of the cosmic banana code which reads in a parameter file
# containing the details of the predictions, data & covariance to be used in
# calculating the likelihood on either a 2D grid, or a 1D line 
# Typically the 2D grid is (Omega_m,S_8) or (Omega_m,sigma_8) grid but this is not set in stone.

import numpy as np
import sys

from Classes_4_Lhd import Get_Input
from Functions_4_Lhd import LogLhd_Gauss, Return_Contours

paramfile = sys.argv[1]
GI = Get_Input(paramfile)
Log_Lhds_Stats, Contours_Stats, Areas_Stats, Constraints_Stats, Log_Lhds_Comb,  Contours_Comb, Areas_Comb, Constraints_Comb = GI.Run_Analysis()



	
	
