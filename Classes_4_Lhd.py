# 02/12/2020, B. Giblin, Edinburgh
# classes to be used in the Cosmic_Banana.py code.

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}                                                                                                           
plt.rc('font', **font)

# Use these to manually create legends
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.interpolate import interp1d

# Emulator classes used if performing an MCMC, emulating at each step.
from Classes_4_GPR import PCA_Class, GPR_Emu

# Class to read analysis information from input parameter file
# e.g. predictions, covariance, data, type of statistic, plotting label
class Get_Input:

	# ------------------------------------------- READ IN PARAMS ------------------------------------------------------
	def __init__(self, paramfile):
		self.paramfile = paramfile
		self.paraminput = open(self.paramfile).read()

		# The following are things used by the lnlike function in MCMC sampling
		# initialised as None, but set once at the start of sampling, to save time
		# by avoiding reading them from file at every step in the chain.
		self.Train_x_4thiscomb           = None		 # x-coords of the emulator training set (e.g. theta [arcmin], k [h/Mpc] )
		self.inTrain_Pred_4thiscomb      = None   	 # (Transformed) Training preds for emulator, stacked for a given combination of stats.
		# --- Following only get used if Perform_PCA is true ---
		self.Train_BFs_4thiscomb         = None		 # Basis functions for this training pred set, stacked for given stats combo.
		self.inTrain_Pred_Mean_4thiscomb = None      # inTrain_Pred_4thiscomb avg'd across the predictions
		self.HPs_4thiscomb               = None      # stacked hyperparameters for this combination of stats.
		

	# --- what stats to use ---
	def Use_Stats(self): 
		return eval(self.paraminput.split('Use_Stats = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def Combine_Stats(self): 
		return eval(self.paraminput.split('Combine_Stats = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	


	# --- 2D or 1D likelihood evaluations ---
	def OneD_TwoD_Or_nD(self):
		return self.paraminput.split('OneD_TwoD_Or_nD = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def x_Res(self):
		return int(self.paraminput.split('x_Res = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def y_Res(self):
		return int(self.paraminput.split('y_Res = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Data nodes ---
	def DataNodesFile(self):
		return self.paraminput.split('DataNodesFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def DataNodesCols(self): 
		return eval(self.paraminput.split('DataNodesCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	


	# --- 2D vs 1D likelihood evaluations ---
	def DataLabel(self):
		return eval( self.paraminput.split('DataLabel = ')[-1].split('#')[0].split('\n')[0] )

	def xLabel(self):
		return eval( self.paraminput.split('xLabel = ')[-1].split('#')[0].split('\n')[0] )

	def yLabel(self):
		return eval( self.paraminput.split('yLabel = ')[-1].split('#')[0].split('\n')[0] )

	def nLabels(self):
		return eval( self.paraminput.split('nLabels = ')[-1].split('#')[0].split('\n')[0] )

	def plot_savename(self):
		output=self.paraminput.split('plot_savename = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if output == '#':
			output='Tmp_plot.png'
		return output

	def savedirectory(self):
		output=self.paraminput.split('savedirectory = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if output == '#':
			output='.'
		return output


	# --- Apply an S8 Prior? ---
	def Apply_S8Prior(self):
		try:
			output=eval( self.paraminput.split('Apply_S8Prior = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output

	def S8_Bounds(self): 
		return eval(self.paraminput.split('S8_Bounds = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Apply the Hartlap factor ---
	def Apply_Hartlap(self):
		try:
			output=eval( self.paraminput.split('Apply_Hartlap = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output	

	# --- MCMC settings ---	
	def nwalkers(self):
		return int(self.paraminput.split('nwalkers = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def burn_steps(self):
		return int(self.paraminput.split('burn_steps = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def real_steps(self):
		return int(self.paraminput.split('real_steps = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

			


	# ------------------------------------------- STATISTIC INFO ------------------------------------------------------

	# ---- Find section of paramfile associated with a given statistic number ---
	def Filter_Stat_Info(self, stat_num):
		return self.paraminput.split('STATISTIC %s ' %stat_num)[-1].split('STATISTIC')[0]


	# --- Predictions per statistic ---	
	def nBins(self, stat_num):
		return int(self.Filter_Stat_Info(stat_num).split('nBins = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def Bins_To_Use(self, stat_num):
		return eval(self.Filter_Stat_Info(stat_num).split('Bins_To_Use = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def PredFile(self, stat_num):
		return self.Filter_Stat_Info(stat_num).split('PredFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def PredIDs(self, stat_num):
		return eval(self.Filter_Stat_Info(stat_num).split('PredIDs = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
	
	def PredCols(self, stat_num): 
		return eval(self.Filter_Stat_Info(stat_num).split('PredCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def PredNodesFile(self, stat_num, string):
		return self.Filter_Stat_Info(stat_num).split('%s = ' %string)[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def PredNodesCols(self, stat_num, string): 
		return eval(self.Filter_Stat_Info(stat_num).split('%s = ' %string)[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def Cols4Plot(self, stat_num): 
		return eval(self.Filter_Stat_Info(stat_num).split('Cols4Plot = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Covariance per statistic ---
	def CovFile(self, stat_num):
		return self.Filter_Stat_Info(stat_num).split('CovFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def CovArea(self, stat_num):
		return eval(self.Filter_Stat_Info(stat_num).split('CovArea = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def SurveyArea(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('SurveyArea = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def Nreal(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('Nreal = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )


	# --- Data per statistic ---
	def DataFile(self, stat_num):
		return self.Filter_Stat_Info(stat_num).split('DataFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def DataCols(self, stat_num): 
		return eval(self.Filter_Stat_Info(stat_num).split('DataCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Plotting each statistic ---
	def PlotLabel(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('PlotLabel = ')[-1].split('#')[0].split('\n')[0] )	

	def PlotColour(self, stat_num):
		return eval(self.Filter_Stat_Info(stat_num).split('PlotColour = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def SmoothContour(self, stat_num):
		try:
			output=eval( self.Filter_Stat_Info(stat_num).split('SmoothContour = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output

	def SmoothScale(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('SmoothScale = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )


	# --- EMULATOR SETTINGS --- #

	def Transform(self, stat_num):
		return self.Filter_Stat_Info(stat_num).split('Transform = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] 

	def Perform_PCA(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('Perform_PCA = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def n_restarts_optimizer(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('n_restarts_optimizer = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def n_components(self, stat_num):
		return eval( self.Filter_Stat_Info(stat_num).split('n_components = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	# --- PRIORS & STARTING MCMC COSMOLOGY PER STATISTIC --- 
	def Priors_Start_MCMC(self, stat_num):
		priorfile=self.Filter_Stat_Info(stat_num).split('Prior_File = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		try:
			priors = np.load( priorfile )
		except FileNotFoundError:
			#print("Using default uniform priors (bounds of training set)")
			nodes= self.LoadPredNodes(stat_num, 'pred') 
			priors = []
			for i in range(nodes.shape[1]):
				priors.append([ nodes[:,i].min(), nodes[:,i].max() ])
			priors = np.array( priors )
		
		# Set starting cosmol to centre of parameter space
		start = np.zeros( priors.shape[0] ) 
		for i in range(nodes.shape[1]):
			start[i] = ( priors[i].min() + priors[i].max() ) /2.
		return priors, start


	# ------------------------------------------- COMBINATION INFO ------------------------------------------------------
	def Filter_Combine_Info(self, stat_num):
		return self.paraminput.split('COMBINATION %s ' %stat_num)[-1].split('COMBINATION')[0]


	# --- Covariance per combination ---
	def CovCombinedFile(self, stat_num):
		return self.Filter_Combine_Info(stat_num).split('CovFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def CovCombinedArea(self, stat_num):
		return eval(self.Filter_Combine_Info(stat_num).split('CovArea = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def SurveyCombinedArea(self, stat_num):
		return eval( self.Filter_Combine_Info(stat_num).split('SurveyArea = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def NrealCombined(self, stat_num):
		return eval( self.Filter_Combine_Info(stat_num).split('Nreal = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	# --- Plotting each combination ---
	def PlotCombinedLabel(self, stat_num):
		return eval( self.Filter_Combine_Info(stat_num).split('PlotLabel = ')[-1].split('#')[0].split('\n')[0] )	

	def PlotCombinedColour(self, stat_num):
		return eval(self.Filter_Combine_Info(stat_num).split('PlotColour = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def SmoothCombinedContour(self, stat_num):
		try:
			output=eval( self.Filter_Combine_Info(stat_num).split('SmoothContour = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output

	def SmoothCombinedScale(self, stat_num):
		return eval( self.Filter_Combine_Info(stat_num).split('SmoothScale = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	# --- Simple string, used to save results for different combinations under different names ---
	def CombName(self, stat_num):
		return self.Filter_Combine_Info(stat_num).split('CombName = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]



	# ------------------------------------------- READING FUNCTIONS ------------------------------------------------------
	def HPs(self, stat_num):
		# Used in cases where the emulator is being executed within an MCMC
		hpfile=self.Filter_Stat_Info(stat_num).split('HPs_File = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] 
		if hpfile == "None" or '---' in hpfile or '#' in hpfile or not hpfile:
			HPs = None
		elif hpfile[-4:] == ".npy":
			HPs = np.load( hpfile )
		else:
			HPs = np.loadtxt( hpfile )
		return HPs

	def LoadPred(self, stat_num):
		nbins = self.nBins(stat_num)
		bins_to_use = self.Bins_To_Use(stat_num)
		predfile = self.PredFile(stat_num)

		if predfile[-4:] == ".npy":
			# Read in pickled file
			y_precut = np.load( predfile )
			# apply cuts 
			y = y_precut[:,bins_to_use]
			x = None # no x-predictions read in
				
		else:
			predIDs = self.PredIDs(stat_num)
			predcols = self.PredCols(stat_num)

			y = np.zeros([ len(predIDs), len(bins_to_use) ])
			for i in range( len(predIDs) ):
				pf = '%s%s%s' %(predfile.split('XXXX')[0],predIDs[i],predfile.split('XXXX')[1])
				x_precut, y_precut = np.loadtxt(pf, usecols=predcols, unpack=True)
				y[i,:] = y_precut[bins_to_use]
			x = x_precut[bins_to_use]

		return x,y

	def LoadData(self, stat_num):
		bins_to_use = self.Bins_To_Use(stat_num)
		datafile = self.DataFile(stat_num)
		cols = self.DataCols(stat_num)
		x_data, y_data = np.loadtxt(datafile, usecols=cols, unpack=True)
		return x_data[bins_to_use], y_data[bins_to_use]


	# ----- Cov. loading & scale selection functions -----
	def ScaleCut_Cov(self, cov, bins_to_use):
		# scroll through the covariance, re-building it with the designated bins
		cov_rebuild = np.zeros([ len(bins_to_use), len(bins_to_use) ])
		for i in range(len(bins_to_use)):
			for j in range(len(bins_to_use)):
				cov_rebuild[i,j] = cov[ bins_to_use[i], bins_to_use[j] ]
		return cov_rebuild


	def LoadCov(self, stat_num):
		bins_to_use = self.Bins_To_Use(stat_num)
		covfile = self.CovFile(stat_num)
		cov = np.load(covfile)

		# extract only the elements of the cov specified by use_bins
		# note this can only omit upper/lower bins, not those in the middle.
		cov = self.ScaleCut_Cov(cov, bins_to_use)
		# area scale:
		covarea = self.CovArea(stat_num)
		surveyarea = self.SurveyArea(stat_num)

		# apply Hartlap correction if specified to:
		if self.Apply_Hartlap():
			Nreal = self.Nreal(stat_num)
			Hfactor = self.Calc_Hartlap(Nreal, cov.shape[0])
		else:
			Hfactor = 1.
		return cov * (covarea / surveyarea) / Hfactor	

	
	def ScaleCut_CovCombined(self, cov, comb_num):
		# identify the statistics used in this combination
		use_stats = self.Combine_Stats()[comb_num-1]

		# pull out the bins_to_use for each statistic in the combination & append them
		sum_nbins = 0 # need to sum lengths of statistics as you scroll through them
		for i in range(len(use_stats)):
			tmp_bins = np.array( self.Bins_To_Use(use_stats[i]) )

			if i==0:
				bins_to_use = tmp_bins
			else:
				bins_to_use = np.append( bins_to_use, tmp_bins+sum_nbins )

			sum_nbins += self.nBins(use_stats[i])

		#print("The bins used in combined covariance %s: "%comb_num, bins_to_use)
		cov = self.ScaleCut_Cov(cov, bins_to_use)
		return cov


	def LoadCovCombined(self, comb_num):
		covfile = self.CovCombinedFile(comb_num)
		cov = np.load(covfile)
		cov = self.ScaleCut_CovCombined(cov, comb_num)

		covarea = self.CovCombinedArea(comb_num)
		surveyarea = self.SurveyCombinedArea(comb_num)

		# apply Hartlap correction if specified to:
		if self.Apply_Hartlap():
			Nreal = self.NrealCombined(comb_num)
			Hfactor = self.Calc_Hartlap(Nreal, cov.shape[0])
		else:
			Hfactor = 1.

		return cov * (covarea / surveyarea) / Hfactor
	


	def LoadPredNodes(self, stat_num, pred_OR_trial):
		if "pred" in pred_OR_trial:
			nodesfile = self.PredNodesFile(stat_num, 'PredNodesFile' )
			cols = self.PredNodesCols(stat_num, 'PredNodesCols' )
		elif "trial" in pred_OR_trial:
			# only will read in Trial Nodes when doing a 1D or 2D likelihood line/grid.
			nodesfile = self.PredNodesFile(stat_num, 'TrialNodesFile' )
			cols = self.PredNodesCols(stat_num, 'TrialNodesCols' )

		nodes = np.loadtxt(nodesfile, usecols=(cols))
		return nodes

		#od_or_td = self.OneD_TwoD_Or_nD()
		#if od_or_td == "2D":
			# 2D analysis, read two numbers in
		#	x_nodes, y_nodes = np.loadtxt(nodesfile, usecols=cols, unpack=True)
		#	return x_nodes, y_nodes

		#elif od_or_td == "1D":
			# 1D analysis, read 1 number in
		#	x_nodes = np.loadtxt(nodesfile, usecols=(cols[0],), unpack=True)
		#	return x_nodes

		#elif od_or_td == "nD":
			# >2 dimensions (performing MCMC likelihood): read all in
		#	nodes = np.loadtxt(nodesfile, usecols=(cols))
		#	return nodes

		#else:
		#	print("OneD_TwoD_Or_nD must be set to either 1D, 2D or nD, not %s. Rectify this!" %od_or_td)
		#	sys.exit(1)


	def LoadNodes4Plot(self, stat_num):
		# identify which columns of the TrialNodesFile will be used for the axes
		# on the contours plot (only used for 1D-line/2D-grid likelihood evaluations)
	
		nodesfile = self.PredNodesFile(stat_num, 'TrialNodesFile' )
		cols = self.Cols4Plot(stat_num)

		od_or_td = self.OneD_TwoD_Or_nD()
		if "2D" in od_or_td:
			# 2D analysis, read two numbers in
			x_nodes, y_nodes = np.loadtxt(nodesfile, usecols=cols, unpack=True)
			return x_nodes, y_nodes

		elif "1D" in od_or_td:
			# 1D analysis, read 1 number in
			x_nodes = np.loadtxt(nodesfile, usecols=(cols[0],), unpack=True)
			return x_nodes

		else:
			print("OneD_TwoD_Or_nD must be set to either 1D, 2D or nD, not %s. Rectify this!" %od_or_td)
			sys.exit(1)



	def LoadDataNodes(self):
		nodesfile = self.DataNodesFile()
		cols = self.DataNodesCols()
		od_or_td = self.OneD_TwoD_Or_nD()
		if "2D" in od_or_td:
			# 2D analysis, read two numbers in
			x_nodes, y_nodes = np.loadtxt(nodesfile, usecols=cols, unpack=True)
			return x_nodes, y_nodes

		elif "1D" in od_or_td:
			# 1D analysis, read 1 number in
			x_nodes = np.loadtxt(nodesfile, usecols=(cols[0],), unpack=True)
			return x_nodes

		elif od_or_td == "nD":
			# >2 dimensions (performing MCMC likelihood): read all in
			nodes = np.loadtxt(nodesfile, usecols=(cols))
			return nodes

		else:
			print("OneD_TwoD_Or_nD must be set to either 1D, 2D or nD, not %s. Rectify this!" %od_or_td)
			sys.exit(1)


	def CombineData(self, use_stats):
		# use_stats is an array containing the numbers of the statistics to combine the data vectors for.
		data = []
		for stat in use_stats:
			d = self.LoadData(stat)[1]
			data = np.append(data, d)
		return data

	def CombinePreds(self, use_stats):
		# use_stats is an array containing the numbers of the statistics to combine the prediction vectors for.
		for stat in use_stats:
			if stat==use_stats[0]:
				pred = self.LoadPred(stat)[1]
			else:
				pred = np.concatenate( (pred,self.LoadPred(stat)[1]), axis=1 )
		return pred


	def Implement_S8Prior(self, LogL, stat_num):
		S8_Bounds = self.S8_Bounds()
		#print("Applying an S8 prior to the likelihood with bounds: ", S8_Bounds)

		if "Emu" in self.OneD_TwoD_Or_nD():
			# If emulating a 2D grid, get the x,y coords from the trial nodes file
			xc, yc = self.LoadNodes4Plot(stat_num)
		else:
			# but if reading in pre-made predictions, get x,y from PredNodesFile
			nodes = self.LoadPredNodes(stat_num, 'pred') 
			xc = nodes[:,0]
			yc = nodes[:,1]		

		# Apply the S8 prior to a likelihood:
		if "Omega" in self.xLabel() and "sigma" in self.yLabel():
			S8 = yc * (xc/0.3)**0.5
		elif "S_8" in self.yLabel():
			S8 = yc

		idx_kill = np.where( np.logical_or(S8<S8_Bounds[0], S8>S8_Bounds[1]) )[0] 
		LogL[idx_kill] = np.inf
		return LogL

	def Calc_Hartlap(self, Nreal, nbin):
		return float(Nreal - nbin - 2.) / (Nreal - 1.)	


	def Marginalise_Over_Dimension(self, LogL, axis, stat_num):

		Lhd = np.exp(-0.5*LogL)

		if "Emu" in self.OneD_TwoD_Or_nD():
			# If emulating a 2D grid, get the x,y coords from the trial nodes file
			xc, yc = self.LoadNodes4Plot(stat_num)
		else:
			# but if reading in pre-made predictions, get x,y from PredNodesFile
			nodes = self.LoadPredNodes(stat_num, 'pred') 
			xc = nodes[:,0]
			yc = nodes[:,1]

		if axis == 0:
			xaxis = np.unique(xc)
			OneD_Lhd = np.zeros(len(Lhd[0,:]))
			for i in range(len(OneD_Lhd)):
				OneD_Lhd[i] = np.sum(Lhd[:,i])

		else:
			xaxis = np.unique(yc)
			OneD_Lhd = np.zeros(len(Lhd[:,0]))
			for i in range(len(OneD_Lhd)):
				OneD_Lhd[i] = np.sum(Lhd[i,:])
		
		def Loop_2_Find_Probability(fraction):
			Sum_Prob=0.
			for i in range(0, int(len(xaxis)*N/2)):
				try:
					Sum_Prob += (interp_function(Mean + i*dx) + interp_function(Mean - i*dx))*dx 
				except ValueError:
					print( "Hit the wall searching for the %s prob-fraction with this statistic/combo of stats beginning with stat %s. " %(fraction,stat_num))
					print( "Only found %.2f of it. Returning this answer." %(Sum_Prob/Total_Prob) )
					return i*dx
				if Sum_Prob > fraction*Total_Prob:
					return i*dx

		# Finding 1D 68/95 percent contours
		Mean = np.sum( OneD_Lhd*xaxis) / np.sum(OneD_Lhd)
		interp_function = interp1d(xaxis, OneD_Lhd, kind='linear')
		N = 1000 # Interpolate on an xaxis with N times higher resolution
		dx = (xaxis[1] - xaxis[0]) / N 

		Total_Prob = np.sum(OneD_Lhd)*dx*N
		Constraints = np.zeros(3)
		Constraints[0] = Mean
		Constraints[1] = Loop_2_Find_Probability(0.68)
		Constraints[2] = Loop_2_Find_Probability(0.95)
		return Constraints


	# ------------------------------------------- FUNCTIONS FOR THE MCMC ------------------------------------------------------
	# log prior
	def lnprior(self, p, comb):
		# scroll through statistics, load priors, see if point is outside allowed range.
		for stat in comb:
			prior_ranges = self.Priors_Start_MCMC(stat)[0]
			for i in range(len(p)):
				if (p[i] < prior_ranges[i,0]) or (p[i] > prior_ranges[i,1]):
					return -np.inf
		return 0.

	# This function is called once at the start of each likelihood sampling
	# to read in, set the emulator Training set + HPs, and apply the relevant transformations
	# for this combination of stats.
	# This is then stored in memory, saving time by avoiding the lnlike func reading from
	# file every step of the chain. 
	def Assemble_TrainPred_and_HPs( self, comb ):
		# temporary arrays to store & stack the info for each stat in this combination.
		HPs_store             = []
		Train_x_store         = []
		inTrain_Pred_store    = []
		Train_Pred_Mean_store = []
		Train_BFs_store       = []

		for stat in comb:
			Train_x, Train_Pred = self.LoadPred(stat)			
			
			# Identify the transformation for this statistic
			if self.Transform(stat) == "log":  
				Train_Pred = np.log( Train_Pred )
			elif "xy" in self.Transform(stat):
				try:
					scale = float( self.Transform(stat).split('xy')[-1] )
				except ValueError:
					scale = 1.
				Train_Pred *= (Train_x*scale)
			else:
				print( "Only log and xyN (x-TIMES-y-TIMES-N) transforms are supported. Not %s. EXITING." %self.Transform(stat) )
				sys.exit()

			if self.Perform_PCA(stat):
				PCAC = PCA_Class(self.n_components(stat))
				Train_BFs, Train_Weights, Train_Recons = PCAC.PCA_BySKL(Train_Pred)
				Train_Pred_Mean = np.mean( Train_Pred, axis=0 )
				inTrain_Pred = np.copy( Train_Weights )
			else:
				inTrain_Pred = np.copy(Train_Pred)
				Train_BFs = []        # dummy to be stored in case you have a mix of stats which do/dont use PCA in this combination.				
				Train_Pred_Mean = []  # ^same. Need these to preserve order of stored BFs and Train Pred Means.
				
			# storing info for each stat in the combination:
			Train_x_store.append( Train_x )
			inTrain_Pred_store.append( inTrain_Pred )
			Train_BFs_store.append( Train_BFs )
			Train_Pred_Mean_store.append( Train_Pred_Mean )
			
			if self.OneD_TwoD_Or_nD() == "nD":
				# then it's doing an MCMC, and should look for a file of HPs which may be saved:
				HPs = self.HPs( stat ) 
				if HPs == None:
					# If not HPs_File is set, need to train emulator once to get these (100 restarts):	
					print( "Running the emulator once with 1000 restarts to get the HPs." )
					Train_Nodes = self.LoadPredNodes(stat, 'pred')
					GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Train_Nodes )	
					_,_,HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, 1000 )	
					print(HPs)	
				HPs_store.append( HPs )

			elif self.OneD_TwoD_Or_nD() == "1DEmu" or self.OneD_TwoD_Or_nD() == "2DEmu":  
				# It's a 2D grid or 1D line of lhood evaluations & we are 
				# emulating all trial predictions here:
				Train_Nodes = self.LoadPredNodes(stat, 'pred')
				Trial_Nodes = self.LoadPredNodes(stat, 'trial')
				GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Trial_Nodes )	
				GP_AVOUT, GP_STDOUT, GP_HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, self.n_restarts_optimizer(stat) )	

				# Un-do the PCA if appropriate
				if self.Perform_PCA(stat):
					PCAC = PCA_Class(self.n_components(stat))
					GP_Pred = PCAC.Convert_PCAWeights_2_Predictions(GP_AVOUT, Train_BFs, Train_Pred_Mean)
				else:
					GP_Pred = GP_AVOUT

				# Un-do the transformation:
				if self.Transform(stat) == "log":  
					GP_Pred = np.exp( GP_Pred )
				elif "xy" in self.Transform(stat):
					GP_Pred = GP_Pred/(Train_x*scale)

				# concatenate the combined statistics
				if stat==comb[0]:
					Trial_Pred_store = GP_Pred
				else:
					Trial_Pred_store = np.concatenate( (Trial_Pred_store, GP_Pred), axis=1 )

		# store everything to be used by the lnlike function
		self.HPs_4thiscomb               = HPs_store
		self.Train_x_4thiscomb           = Train_x_store	 
		self.inTrain_Pred_4thiscomb      = inTrain_Pred_store 	 
		# --- Following only get used if Perform_PCA is true ---
		self.Train_BFs_4thiscomb         = Train_BFs_store  		 
		self.inTrain_Pred_Mean_4thiscomb = Train_Pred_Mean_store 

		if self.OneD_TwoD_Or_nD() == "1DEmu" or self.OneD_TwoD_Or_nD() == "2DEmu":   
			return Trial_Pred_store

		return



	# log likelihood (executing emulator at each step). 
	def lnlike(self, p, cov, data, comb):

		# scroll through the statistics in this combination, 
		# read in the predictions (emulator needs these even when trained),
		# and emulate prediction for this cosmology.
		# start by checking the TrainPred has been correctly set for this combo of stats:
		if self.inTrain_Pred_4thiscomb == None:
			self.Assemble_TrainPred_and_HPs( comb )


		GP_Pred_All = np.ones([]) # Store the emulated predictions for each stat used in the combination.
		count = 0                 # count increments through statistics
		for stat in comb:
			# Grab the important data from memory needed for the likelihood evaluation
			HPs             = self.HPs_4thiscomb[count]
			Train_x         = self.Train_x_4thiscomb[count]	 
			inTrain_Pred    = self.inTrain_Pred_4thiscomb[count]       
			Train_BFs       = self.Train_BFs_4thiscomb[count]          		 
			Train_Pred_Mean = self.inTrain_Pred_Mean_4thiscomb[count]
		
			# Run the emulator		
			GPR_Class = GPR_Emu( self.LoadPredNodes(stat, 'pred'), inTrain_Pred, np.zeros_like(inTrain_Pred), p.reshape(1,-1) )	
			GP_AVOUT, GP_STDOUT, GP_HPs = GPR_Class.GPRsk(HPs, None, self.n_restarts_optimizer(stat) )	

			# Un-do the PCA if appropriate
			# Output is shaped (1,n_components) --> need to select [0,:]
			if self.Perform_PCA(stat):
				PCAC = PCA_Class(self.n_components(stat))
				GP_Pred = PCAC.Convert_PCAWeights_2_Predictions(GP_AVOUT, Train_BFs, Train_Pred_Mean)[0,:]
			else:
				GP_Pred = GP_AVOUT[0,:]

			# Un-do the transformation:
			if self.Transform(stat) == "log":  
				GP_Pred = np.exp( GP_Pred )
			elif "xy" in self.Transform(stat):
				try:
					scale = float( self.Transform(stat).split('xy')[-1] )
				except ValueError:
					scale = 1.
				GP_Pred = GP_Pred/(Train_x*scale)

			# Combine the predictions
			GP_Pred_All = np.append( GP_Pred_All, GP_Pred ) 
			
			# increment statistics count.
			count +=1

		GP_Pred_All = np.delete( GP_Pred_All, 0 )  # get rid of first element (comes from initialisation) & un-do log transform
		LnLike = -0.5 * np.dot( np.transpose(data - GP_Pred_All), np.dot(np.linalg.inv(cov), (data - GP_Pred_All)  ))
		return LnLike


	# log posterior
	def lnprob(self, p, cov, data, comb):
		lp = self.lnprior(p, comb)
		return lp + self.lnlike(p, cov, data, comb) if np.isfinite(lp) else -np.inf

	def Run_MCMC(self, comb_num, comb):

		import emcee
		import time

		# Load the data vector and covariance to be used in this sampling
		cov = self.LoadCovCombined(comb_num)
		data = self.CombineData( comb )
		
		p = self.Priors_Start_MCMC(comb[0])[1]   # Load the starting cosmology (middle of param space)
		ndim = len(p)
		p0 = [p + 1e-2*np.random.randn(ndim) for i in range(self.nwalkers())]		# starting position
		sampler = emcee.EnsembleSampler(self.nwalkers(), ndim, self.lnprob, args=[cov, data, comb])

		t0 = time.time()
		print( "Running burn in of %s steps per walker...." %self.burn_steps() )
		burn_output = sampler.run_mcmc(p0, self.burn_steps() )
		p0 = burn_output.coords
		lnp = burn_output.log_prob
		sampler.reset()

		t1 = time.time()
		print( "First burn-in took %.1f minutes. Running 2nd burn-in..." %((t1-t0)/60.) )
		# set new start point to be a tiny gauss ball around position of whatever walker reached max posterior during burn-in
		p = p0[np.argmax(lnp)]
		p0 = [p + 1.e-2* p * np.random.randn(ndim) for i in range(self.nwalkers())]
		burn2_output = sampler.run_mcmc(p0, self.burn_steps() )
		p0 = burn2_output.coords
		lnp = burn2_output.log_prob
		sampler.reset()

		t2 = time.time()
		print( "Second burn-in took %.1f minutes. Running the MCMC proper with %s steps per walker" %(((t2-t1)/60.), self.real_steps() ) )
		sampler.run_mcmc(p0, self.real_steps() )
		# sampler has an attribute called chain that is 3D: nwalkers * real_steps * ndim in dimensionality
		# Following line turns it into 2D: (nwalkers*real_steps) * ndim. 
		# As if there was only one walker.
		samples = sampler.chain[:, :, :].reshape((-1, ndim))
		t3 = time.time()
		print( "Finished. Main MCMC took %.1f minutes. The whole MCMC took %.1f minutes." %( ((t3-t2)/60.), ((t3-t0)/60.) ) )

		# Just in case the parameter file specifies MULTIPLE combinations of stats, we need to RESET
		# the following variables, which stored data specific to each combination of statistics in memory.
		self.Train_x_4thiscomb           = None		 
		self.inTrain_Pred_4thiscomb      = None   	 
		self.Train_BFs_4thiscomb         = None		 
		self.inTrain_Pred_Mean_4thiscomb = None      
		self.HPs_4thiscomb               = None     	 

		return samples



	def Master_Run_MCMC(self, comb_num, comb):
		print( "---------------- RUNNING STATS COMBO %s "%comb_num, comb, "---------------------------------------------- " )
		if not os.path.exists(self.savedirectory()):
	 		os.makedirs(self.savedirectory())

		savename = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_%s" %(self.savedirectory(), self.SurveyCombinedArea(comb_num), 
																				  self.nwalkers(), self.real_steps(), self.CombName(comb_num) )
		samples = self.Run_MCMC(comb_num, comb)
		np.save( savename, samples )
		#self.Plot_MCMC_Lhd(samples, savename)

		print( "---------------- FINISHED STATS COMBO %s "%comb_num, comb, "---------------------------------------------- " )
		return samples, savename



	# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
	def Plot_2D_Lhd(self, Log_Lhds_Stats, Contours_Stats, 
							  Log_Lhds_Comb, Contours_Comb):
		# Plot a 2D likelihood grid
		Use_Stats = self.Use_Stats()
		LW = 4 # Linewidths
		handles = []
		
		plt.figure(figsize=(11,9))

		# scroll through the statistics to plot contours for
		for i in range( len(Use_Stats) ):

			if "Emu" in self.OneD_TwoD_Or_nD():
				# If emulating a 2D grid, get the x,y coords from the trial nodes file
				xc, yc = self.LoadNodes4Plot(Use_Stats[i])
			else:
				# but if reading in pre-made predictions, get x,y from PredNodesFile
				nodes = self.LoadPredNodes(Use_Stats[i], 'pred') 
				xc = nodes[:,0]
				yc = nodes[:,1]

			plt.contour(Log_Lhds_Stats[i], [Contours_Stats[i][0], Contours_Stats[i][1]], 
						origin='lower', extent = [xc.min(), xc.max(), yc.min(), yc.max()], linewidths=LW, 
						colors=self.PlotColour(Use_Stats[i]) )
			handles.append( mlines.Line2D([],[],color=self.PlotColour(Use_Stats[i]), linewidth=LW, label=self.PlotLabel(Use_Stats[i])) ) 


		# scroll through the combinations of statistics to plot contours for
		Combine_Stats = self.Combine_Stats()
		for i in range( len(Combine_Stats) ):

			if "Emu" in self.OneD_TwoD_Or_nD():
				# If emulating a 2D grid, get the x,y coords from the trial nodes file
				xc, yc = self.LoadNodes4Plot(Combine_Stats[i][0]) # read cosmol coords for the 1st stat of the i'th combination
			else:
				# but if reading in pre-made predictions, get x,y from PredNodesFile
				nodes = self.LoadPredNodes(Combine_Stats[i][0], 'pred') 
				xc = nodes[:,0]
				yc = nodes[:,1]

			
			plt.contour(Log_Lhds_Comb[i], [Contours_Comb[i][0], Contours_Comb[i][1]], 
						origin='lower', extent = [xc.min(), xc.max(), yc.min(), yc.max()], linewidths=LW, 
						colors=self.PlotCombinedColour(i+1) )
			handles.append( mlines.Line2D([],[],color=self.PlotCombinedColour(i+1), linewidth=LW, label=self.PlotCombinedLabel(i+1)) ) 

		# Plot the truth			
		x_data, y_data = self.LoadDataNodes()
		datastar = plt.scatter(x_data, y_data, marker='*', color='yellow', edgecolor='black', s=400, zorder=2, label=self.DataLabel() )

		plt.xlabel( self.xLabel() )
		plt.ylabel( self.yLabel() )	
		plt.legend(handles=handles, loc='upper right', frameon=False, scatterpoints=1)
		plt.savefig(self.plot_savename() )
		plt.show()
		return


	def Plot_MCMC_Lhd(self, samples, savename):
	
		import matplotlib.gridspec as gridspec
		import corner

		steps2plot = np.linspace(0., samples.shape[0]-1, samples.shape[0]) 
		l = len(self.nLabels())

		fig = plt.figure(figsize = (8,16))
		gs1 = gridspec.GridSpec(l, 1)
		for i in range(l):
			# i = i + 1 # grid spec indexes from 0
			ax1 = plt.subplot(gs1[i])
			ax1.plot(steps2plot, samples[:,i], color='magenta', linewidth=1)
			if i != (l-1):
				ax1.set_xticklabels([])
			ax1.set_ylabel(self.nLabels()[i])
			ax1.set_xlabel(r'Number of steps')

		#gs1.update(wspace=0., hspace=0.) # set the spacing between axes.
		#fig.subplots_adjust(hspace=0, wspace=0)
		plt.savefig('%s_Steps.png'%savename)
		#plt.show()

		fig = corner.corner(samples, labels=self.nLabels(), levels=(0.68,0.95), show_titles=True, title_fmt='.3f', truths=self.LoadDataNodes())
		if savename != None and savename != '':
			plt.savefig('%s_Contours.png' %savename)

		#plt.show()
		return 


	def Plot_MCMC_Lhd_Multiple(self, savename): 
	
		import corner
		import matplotlib.lines as mlines

		font = {'family' : 'serif',
	        'weight' : 'normal',
	        'size'   : 20}                                                                                                           
		plt.rc('font', **font)
		lw = 2
		max_n_ticks = 3

		# Scroll through combinations of statistics, read in samples & plot their contours.
		constraints = [] # save and return the mean & asymmetric 68% confidence regions.
		handles = []
	
		# If plot_limits set in parameter file, read them in:
		try:
			limits = eval( self.paraminput.split('Plot_Limits = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			limits = None

		print("The plot limits have been set to: ", limits) 


		for i in range( len(self.Combine_Stats()) ):
			comb_num = i+1
			# sample_name matches the one used in Master_Run_MCMC
			sample_name = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_%s.npy" %(self.savedirectory(), self.SurveyCombinedArea(comb_num), 
																					self.nwalkers(), self.real_steps(), self.CombName(comb_num) )
			samples = np.load( sample_name )
			
			if limits == None:
				# If no limits were read in, set them to the range of the first set of samples.
				limits = []
				for j in range(samples.shape[1]):
					limits.append([ samples[:,j].min(), samples[:,j].max() ])  
		

			if i==0:
				# Then it's the first set of samples, establish fig:
				fig = corner.corner(samples, labels=self.nLabels(), range=limits,
						plot_contours=True, plot_density=False, plot_datapoints=False,
						smooth=1, levels=(0.68,0.95), truths=self.LoadDataNodes(), truth_color='black', 
						contour_kwargs={'colors':[self.PlotCombinedColour( comb_num )], 'linewidths':lw},
						hist_kwargs={'color':[self.PlotCombinedColour( comb_num )], 'linewidth':lw},
						max_n_ticks=max_n_ticks)

			else:
				# sequential set of contours - overplot them
				fig = corner.corner(samples, labels=self.nLabels(), range=limits,
						plot_contours=True, plot_density=False, plot_datapoints=False,
						smooth=1, levels=(0.68,0.95), truths=self.LoadDataNodes(),
						contour_kwargs={'colors':[self.PlotCombinedColour( comb_num )], 'linewidths':lw},
						hist_kwargs={'color':[self.PlotCombinedColour( comb_num )], 'linewidth':lw},
						max_n_ticks=max_n_ticks, fig=fig)

			handles.append( mlines.Line2D([], [], color=self.PlotCombinedColour( comb_num ), label=self.PlotCombinedLabel( comb_num )) )

			# extract and save the constraints:
			tmp_constraints = np.empty([ samples.shape[1], 3 ])
			for j in range( samples.shape[1] ):
				mean = corner.quantile( samples[:,j], [0.5] )[0]
				upper = corner.quantile( samples[:,j], [0.84] )[0] - mean
				lower = mean - corner.quantile( samples[:,j], [0.16] )[0] 
				tmp_constraints[j] = np.array([ mean, upper, lower ])
			constraints.append( tmp_constraints )
		
		fig.set_size_inches((12,10))
		plt.legend(handles=handles, bbox_to_anchor=(0., 2.2, 1.0, .0), loc=4)

		if savename == None:
			# If no savename given use this one as default.
			# Matches the format of the sample_name, specified above and originally in Master_Run_MCMC.
			savename = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_AllComb_Contours.png" %(self.savedirectory(), self.SurveyCombinedArea(1), 
																						self.nwalkers(), self.real_steps() )		
		plt.savefig(savename)
		plt.show()
		return constraints

	


	# ------------------------------------------- OVERALL RUN CODE ------------------------------------------------------	
	def Run_Analysis(self):
		# Overall function to scroll through the designated statistics and combinations of statistics,
		# computing the 2D/1D likelihoods for the specified data vectors
		# and produce a plot at the end.		

		from Functions_4_Lhd import LogLhd_Gauss, Return_Contours, Return_Contour_Areas

		# Scroll through the statistics we're reading in & calculating likelihoods for
		Use_Stats = self.Use_Stats()
		Log_Lhds_Stats = []    # store natural log of likelihoods per statistic
		Contours_Stats = []    # store 1&2sigma contour levels per statistic
		Areas_Stats = []       # store the areas of the 1&2 sigma contours per statistic
		Constraints_Stats = [] # store the 1D constraints on the x&y axes: [(Mean,1sig,2sig)_x,(Mean,1sig,2sig)_y] 

		for stat in Use_Stats:
			print("Producing likelihood for statistic number %s of " %stat, Use_Stats)

			if "Emu" in self.OneD_TwoD_Or_nD():
				# Read in a training and trial set, and emulate the predictions	
				preds = self.Assemble_TrainPred_and_HPs( [stat] )
			else:
				# Read in some pre-made predictions
				preds = self.LoadPred(stat)[1] 
			print(" shape of predictions is ", preds.shape )

			# Load the cov, data & calc the likelihood on the grid
			cov = self.LoadCov(stat)
			data = self.LoadData(stat)[1]
			LogL = LogLhd_Gauss(preds, data, cov)
			# apply an S8 prior to the likeilhood if specified.
			if self.Apply_S8Prior():
				LogL = self.Implement_S8Prior(LogL, stat)

			if "2D" in self.OneD_TwoD_Or_nD():
				# reshape the likelihood to 2D
				LogL = np.reshape(LogL, (-1, self.x_Res() ))

			if self.SmoothContour(stat):
				SS = self.SmoothScale(stat)
				print("Smoothing contour for statistic %s with sigma=%s [pxls]" %(stat, SS))
				from scipy.ndimage import gaussian_filter as gauss
				LogL = gauss(LogL, sigma=[SS,SS])


			# Store the 68% & 95% contour areas, and the Log-likelihood
			contours = Return_Contours(LogL)
			Contours_Stats.append( contours )
			Areas_Stats.append( Return_Contour_Areas(LogL, contours) )
			Log_Lhds_Stats.append( LogL )

			# get and store the 1D x & y constraints
			x_constraints = self.Marginalise_Over_Dimension(LogL, 0, stat)
			y_constraints = self.Marginalise_Over_Dimension(LogL, 1, stat)  
			Constraints_Stats.append( np.vstack((x_constraints,y_constraints)) )

		# Now scroll through the combinations of statistics, if any
		Combine_Stats = self.Combine_Stats()
		Log_Lhds_Comb = []    # store natural log of likelihoods per combination of statistics
		Contours_Comb = []    # store 1&2sigma contour levels per combination of statistics
		Areas_Comb = []       # store the areas of the 1&2 sigma contours per combination of statistics
		Constraints_Comb = [] # store the 1D constraints on the x&y axes: [(Mean,1sig,2sig)_x,(Mean,1sig,2sig)_y] 

		for i in range(len(Combine_Stats)):
			print("Producing likelihood for %s'th combination of statistics "%(i+1), Combine_Stats[i])

			if "Emu" in self.OneD_TwoD_Or_nD():
				# Read in a training and trial set, and emulate the predictions	
				preds = self.Assemble_TrainPred_and_HPs( Combine_Stats[i] )
			else:
				# Read in some pre-made predictions
				preds = self.CombinePreds( Combine_Stats[i] )
			print(" shape of predictions is ", preds.shape )

			cov = self.LoadCovCombined(i+1)
			data = self.CombineData( Combine_Stats[i] )
			LogL = LogLhd_Gauss(preds, data, cov)
			if self.Apply_S8Prior():
				LogL = self.Implement_S8Prior(LogL, Combine_Stats[i][0])

			if "2D" in self.OneD_TwoD_Or_nD():
				# reshape the likelihood to 2D
				LogL = np.reshape(LogL, (-1, self.x_Res() ))

			if self.SmoothCombinedContour(i+1):
				SS = self.SmoothCombinedScale(i+1)
				print("Smoothing contour for combination %s with sigma=%s [pxls]" %(i+1, SS))
				from scipy.ndimage import gaussian_filter as gauss
				LogL = gauss(LogL, sigma=[SS,SS])

			# Store the 68% & 95% contour areas, and the Log-likelihood
			contours = Return_Contours(LogL)
			Contours_Comb.append( contours )
			Areas_Comb.append( Return_Contour_Areas(LogL, contours) )
			Log_Lhds_Comb.append( LogL )

			# get and store the 1D x & y constraints
			x_constraints = self.Marginalise_Over_Dimension(LogL, 0, Combine_Stats[i][0])
			y_constraints = self.Marginalise_Over_Dimension(LogL, 1, Combine_Stats[i][0])  
			Constraints_Comb.append( np.vstack((x_constraints,y_constraints)) )

	
		# Plot the likelihood
		self.Plot_2D_Lhd(Log_Lhds_Stats, Contours_Stats, 
					     Log_Lhds_Comb,  Contours_Comb)

		return Log_Lhds_Stats, Contours_Stats, Areas_Stats, Constraints_Stats, Log_Lhds_Comb,  Contours_Comb, Areas_Comb, Constraints_Comb



	def Run_Analysis_MCMC(self):
		# Overall function to scroll through the designated statistics and combinations of statistics,
		# running an MCMC for each one, executing the emulator at each step of the chain.
		# Make a nice MCMC plot at the end.

		# Scroll through the statistics we're reading in & calculating likelihoods for
		Use_Stats = self.Use_Stats()
		Samples_Stats = []
		Savename_Stats = []
		for i in range(len(Use_Stats)):
			samples, savename = self.Master_Run_MCMC( Use_Stats[i], [Use_Stats[i]] ) 
			Samples_Stats.append(samples)
			Savename_Stats.append(savename)

		# Now scroll through the combinations of statistics, if any
		Combine_Stats = self.Combine_Stats()
		Samples_Comb = []
		Savename_Comb = []
		for i in range(len(Combine_Stats)):
			samples, savename = self.Master_Run_MCMC(i+1, Combine_Stats[i]) 
			Samples_Comb.append( samples )
			Savename_Comb.append( savename )

		return Samples_Stats, Savename_Stats, Samples_Comb, Savename_Comb





