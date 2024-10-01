# 02/12/2020, B. Giblin, Edinburgh
# classes to be used in the Cosmic_Banana.py code.

import numpy as np

def LogLhd_Gauss(pred, data, cov):
	# calculates the natural log of the likelihood 
	# given a 2D array of predictions (each row being a different prediction)
	# a data vector of equal length and a covariance.
	# Uses a Gaussian likelihood and assumes all scale cuts have already been made
	# on pred, data & cov:

	LogL = np.zeros( pred.shape[0] )
	for i in range(pred.shape[0]):
		LogL[i] = np.dot( np.transpose(data - pred[i]), np.dot( np.linalg.inv(cov), (data - pred[i]) ) )
	return LogL



def Find_Height_At_Contour_Area(Grid, Area):
	# This Function finds the height corresponding to a certain area of the contour
	# Grid passed must be a LIKELIHOOD not a chi2
	# Area must be given as a fraction of the whole area.
	# 'Grid' can be 2D array or 1D array, either works.

	sorted_L = np.sort(np.ravel(Grid)) 						# flatten to 1D array and put in order  	
	sorted_L = sorted_L[::-1] 								# put max elements first - want to start at peak of likelihood.
	Total_Prob = np.sum(sorted_L)

	# scroll down from the peak of the distribution: adding the heights = adding the volume
	# since each column has the same base area. Stop when you've reached queried Area
	for i in range(len(sorted_L)):
		#print(" found this proportion of area: %.1f " %(np.sum(sorted_L[:i]) / Total_Prob) )
		if np.sum(sorted_L[:i]) / Total_Prob > Area:
			Height = sorted_L[i-1]                      
			break
	return Height

def Return_Contours(LogL):
	# ... exp[-0.5*LogL_X ] makes the posterior; so factors -0.5 and -2 necessary below.
	contour = np.zeros(2)
	contour[0] = -2.*np.log(Find_Height_At_Contour_Area(np.exp(-0.5*LogL), 0.68))	
	contour[1] = -2.*np.log(Find_Height_At_Contour_Area(np.exp(-0.5*LogL), 0.95))
	return	contour


def Find_Area_Within_Contour_Height(LogL_Grid, Height):
	sorted_L = np.ravel(LogL_Grid) 
	Total_Area = float(len(sorted_L))								
	Qualifying_Heights = sorted_L[ np.where(sorted_L <= Height)[0] ]
	return float(len(Qualifying_Heights)) / Total_Area

def Return_Contour_Areas(LogL_Grid, contour):
	Area = np.zeros(2)
	Area[0] = Find_Area_Within_Contour_Height(LogL_Grid, contour[0])
	Area[1] = Find_Area_Within_Contour_Height(LogL_Grid, contour[1])
	return Area






