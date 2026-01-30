# 02/12/2020, B. Giblin, Edinburgh
# classes to be used in the Cosmic_Banana.py code.

import numpy as np
import sys
import os

from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['ps.useafm'] = False #True
rcParams['pdf.use14corefonts'] = False
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

# This function suppresses the warnings from GP training
def warn(*args, **kwargs):
	pass
import warnings
#warnings.warn = warn
# maybe we should let the warnings through after all?

# Class to read analysis information from input parameter file
# e.g. predictions, covariance, data, type of statistic, plotting label
class Get_Input:

	# ------------------------------------------- READ IN PARAMS ------------------------------------------------------
	def __init__(self, pfile_stats, pfile_combs, pfile_sys = None):
		self.pfile_stats = pfile_stats
		self.pfile_combs = pfile_combs
		self.pfile_sys = pfile_sys
		self.pinput_stats = open(self.pfile_stats).read()
		self.pinput_combs = open(self.pfile_combs).read()

		# Read sys file(s) if provided
		if type(pfile_sys) != type(None):
			self.pfile_sys = pfile_sys.split(',') # turn input into array of strings (1 elem long if just 1 psys file)
			self.pinput_sys = [] # to store string contents of psys files
			for ps in range(len(self.pfile_sys)): self.pinput_sys.append( open(self.pfile_sys[ps]).read() )

		# The following are things used by the lnlike function in MCMC sampling
		# initialised as None, but set once at the start of sampling, to save time
		# by avoiding reading them from file at every step in the chain.
		self.Train_x_4thiscomb 		 = None		 # x-coords of the emulator training set (e.g. theta [arcmin], k [h/Mpc] )
		self.inTrain_Pred_4thiscomb 	 = None   	 # (Transformed) Training preds for emulator, stacked for a given combination of stats.
		# --- Following only get used if Perform_PCA is true ---
		self.Train_BFs_4thiscomb 	 = None		 # Basis functions for this training pred set, stacked for given stats combo.
		self.inTrain_Pred_Mean_4thiscomb = None      # inTrain_Pred_4thiscomb avg'd across the predictions
		self.HPs_4thiscomb 		 = None      # stacked hyperparameters for this combination of stats.
		# --- Only gets used if Transform is set to Norm ---
		self.Train_scaler_4thiscomb	 = None

		# Same but now to store info related to the Systematic predictions:
		self.inSys_Pred_4thiscomb 	= None 
		self.Sys_BFs_4thiscomb 		= None	
		self.inSys_Pred_Mean_4thiscomb 	= None 
		self.Sys_HPs_4thiscomb 		= None 
		self.Sys_scaler_4thiscomb	= None
		self.Sys_cov_4thiscomb		= None

		# And these arrays are used generally in the MCMC:
		self.Priors_Type		= None  # array of strings ("uniform" or "gaussian") of length Dim
		self.Priors 			= None  # if "uniform" this is [[upper,lower],...
							# if "gaussian", [[mean,stdev],....	


	# --- what stats to use ---
	def Use_Stats(self): 
		return eval(self.pinput_stats.split('Use_Stats = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def Combine_Stats(self): 
		return eval(self.pinput_combs.split('Combine_Stats = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	


	# --- 2D or 1D likelihood evaluations ---
	def OneD_TwoD_Or_nD(self):
		return self.pinput_combs.split('OneD_TwoD_Or_nD = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def x_Res(self):
		return int(self.pinput_stats.split('x_Res = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def y_Res(self):
		return int(self.pinput_stats.split('y_Res = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Data nodes ---
	def DataNodesFile(self):
		return self.pinput_combs.split('DataNodesFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def DataNodesCols(self): 
		return eval(self.pinput_combs.split('DataNodesCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	


	# --- 2D vs 1D likelihood evaluations ---
	def DataLabel(self):
		return eval( self.pinput_combs.split('DataLabel = ')[-1].split('#')[0].split('\n')[0] )

	def xLabel(self):
		return eval( self.pinput_combs.split('xLabel = ')[-1].split('#')[0].split('\n')[0] )

	def yLabel(self):
		return eval( self.pinput_combs.split('yLabel = ')[-1].split('#')[0].split('\n')[0] )

	def nLabels(self):
		return eval( self.pinput_combs.split('nLabels = ')[-1].split('#')[0].split('\n')[0] )

	def Plot_Dims(self):
		output = self.pinput_combs.split('Plot_Dims = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if output == '#':
			return None # no plot dimensions provided
		else:
			return eval(output) 

	def plot_savename(self):
		output=self.pinput_combs.split('plot_savename = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if output == '#':
			output='Tmp_plot.png'
		return output


	def savedirectory(self):
		output=self.pinput_combs.split('savedirectory = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if output == '#':
			output='.'
		return output


	# --- Apply an S8 Prior? ---
	def Apply_S8Prior(self):
		try:
			output=eval( self.pinput_combs.split('Apply_S8Prior = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output

	def S8_Bounds(self): 
		return eval(self.pinput_combs.split('S8_Bounds = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Apply the Hartlap factor ---
	def Apply_Hartlap(self):
		try:
			output=eval( self.pinput_combs.split('Apply_Hartlap = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output	

	# --- MCMC settings ---	
	def nwalkers(self):
		return int(self.pinput_combs.split('nwalkers = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def burn_steps(self):
		return int(self.pinput_combs.split('burn_steps = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def real_steps(self):
		return int(self.pinput_combs.split('real_steps = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

			


	# ------------------------------------------- STATISTIC INFO ------------------------------------------------------

	# ---- Find section of paramfile associated with a given statistic number ---
	# ---- this works for both the stats & sys input paramfiles (determined by pinput) --- 
	def Filter_Stat_Info(self, pinput, stat_num):
		return pinput.split('STATISTIC %s ' %stat_num)[-1].split('STATISTIC')[0]


	# --- Predictions per statistic ---	
	def nBins(self, stat_num):
		return int(self.Filter_Stat_Info(self.pinput_stats, stat_num).split('nBins = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def Bins_To_Use(self, stat_num):
		return eval(self.Filter_Stat_Info(self.pinput_stats, stat_num).split('Bins_To_Use = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def PredFile(self, pinput, stat_num):
		return self.Filter_Stat_Info(pinput, stat_num).split('PredFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def PredIDs(self, pinput, stat_num):
		return eval(self.Filter_Stat_Info(pinput, stat_num).split('PredIDs = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
	
	def PredCols(self, pinput, stat_num): 
		return eval(self.Filter_Stat_Info(pinput, stat_num).split('PredCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	

	def PredNodesFile(self, pinput, stat_num, string):
		return self.Filter_Stat_Info(pinput, stat_num).split('%s = ' %string)[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def PredNodesCols(self, pinput, stat_num, string): 
		return eval(self.Filter_Stat_Info(pinput, stat_num).split('%s = ' %string)[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def Cols4Plot(self, stat_num): 
		return eval(self.Filter_Stat_Info(self.pinput_stats, stat_num).split('Cols4Plot = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Data per statistic ---
	def DataFile(self, stat_num):
		return self.Filter_Stat_Info(self.pinput_stats, stat_num).split('DataFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def DataCols(self, stat_num): 
		return eval(self.Filter_Stat_Info(self.pinput_stats, stat_num).split('DataCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --- Plotting each statistic ---
	# NB: NEXT 2 FUNCNS REDUNDANT? ALL HANDLED BY COMBINATIONS FILE?
	def SmoothContour(self, stat_num):
		try:
			output=eval( self.Filter_Stat_Info(self.pinput_stats, stat_num).split('SmoothContour = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			output=False
		return output

	def SmoothScale(self, stat_num):
		return eval( self.Filter_Stat_Info(self.pinput_stats, stat_num).split('SmoothScale = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )


	# ----- FOLLOWING USED FOR SYS PARAMS SPECIFICALLY ----- #

	# determine how many nuisance params associated with each sys; if not specified in sys paramfile, default is 1.
	# note this is not a function of statistic (global variable per sys), so not using Filter_Stat_Info func'n.
	def Nparams_sys(self, pinput): 
		try:
			output = int(pinput.split('Nparams_sys = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except ValueError:
			output = 1
		return output

	# if multiple nuis params, this cov describes their corr'n. Must be provided if Nparams_sys>1
	def SysCov(self, pinput): 
		cfile = pinput.split('SysCovFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		return np.load(cfile)


	# So far only called for psys files (in Priors_Start_MCMC); identifies if it's uniform (default) or gaussian priors
	def PriorString(self, pinput):
		output = str(pinput.split('PriorString = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]).strip()
		#print("PriorString is....", output)
		if output == 'corr-gaussian':
			return output
		elif "gaussian" in output or "Gaussian" in output:
			return "gaussian"
		else:
			return "uniform"

	# Only called (in Priors_Start_MCMC) if PriorString is "gaussian"; reads in [mean,stdev] per sampled param.
	def PriorVals(self, pinput): 
		try:
			output = eval(pinput.split('PriorVals = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except SyntaxError:
			output = None 
		return output



	# If Nparams_Sys (above) > 1, which params to use for each stat? Read in this info here:
	def which_params(self, pinput, stat_num): 
		try:
			output = eval(self.Filter_Stat_Info(pinput, stat_num).split('which_params = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except SyntaxError:
			output = None 
		return output



	# --- EMULATOR SETTINGS --- #

	def Model(self, pinput, stat_num):
		return self.Filter_Stat_Info(pinput, stat_num).split('Model = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] 


	def Transform(self, pinput, stat_num):
		return self.Filter_Stat_Info(pinput, stat_num).split('Transform = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] 

	def Perform_PCA(self, pinput, stat_num):
		return eval( self.Filter_Stat_Info(pinput, stat_num).split('Perform_PCA = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def n_restarts_optimizer(self, pinput, stat_num):
		return eval( self.Filter_Stat_Info(pinput, stat_num).split('n_restarts_optimizer = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	def n_components(self, stat_num):
		return eval( self.Filter_Stat_Info(self.pinput_stats, stat_num).split('n_components = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0] )

	# --- PRIORS & STARTING MCMC COSMOLOGY PER STATISTIC --- 
	def Priors_Start_MCMC(self, stat_num):
		# stat_num is just used to get training_set nodes: 
		# these will be bounds of the prior (if uniform) unless otherwise specified
		print("The input files are as follows:")
		print("stats file: ", self.pfile_stats)
		print("combinations file: ", self.pfile_combs)
		print("sys file(s): ", self.pfile_sys)
		print("The chain will be saved with tagline: ", self.CombName(1))
		print("------------------------------------------------")		

		nodes = self.LoadPredNodes(self.pinput_stats, stat_num, 'pred') 
		priorfile = self.pinput_combs.split('Prior_File = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		priors_type = []
		try:
			priors = np.load( priorfile )
			for i in range(nodes.shape[1]): priors_type.append( self.PriorString(self.pinput_combs) )
			
		except FileNotFoundError:
			priors = []
			for i in range(nodes.shape[1]):
				priors_type.append( self.PriorString(self.pinput_combs) )
				# if PriorString set to gaussian, read in [mean,stdev]
				# else set to the bounds of the training set
				if priors_type[-1] == "gaussian":
					priors.append( self.PriorVals(self.pinput_combs) )
				else:
					priors.append([ nodes[:,i].min(), nodes[:,i].max() ])
				

			# Now do the same with the sys-files if these have been submitted:
			if type(self.pfile_sys) != type(None):
				for ps in range(len(self.pinput_sys)):  # cycle through sys-files
					# check if PriorVals (for all params) are specified in sys-file:
					# this will be the case if gaussian prior, or using non-default uniform
					pvals = self.PriorVals(self.pinput_sys[ps])
					
					# How many sys params?
					Np_sys = self.Nparams_sys(self.pinput_sys[ps])
					for i in range(Np_sys):
						priors_type.append( self.PriorString(self.pinput_sys[ps]) )
						if type(pvals) != type(None):
							priors.append( pvals[i] )
						else:
							nodes_sys = self.LoadPredNodes(self.pinput_sys[ps], stat_num, 'pred') 
							priors.append([ nodes_sys[:,i].min(), nodes_sys[:,i].max() ])

			priors = np.array( priors )
		
		# Set starting point to centre of parameter space (if uniform) or peak of prior (if gaussian)
		if priors_type[0] == "corr-gaussian":
			print(".......... USING CORR'D GAUSS PRIOR ACROSS PARAMS .......... :")
			start = np.array([ 0.2905, 0.8360, 0.6898, -1.000 ]) # SLICS! NOTE, SHOULD BE UPDATED.
		else:
			print(".......... SET PRIORS TO .......... :")
			print(" | PARAM | TYPE | PARAMS | START POS |" )
			start = np.zeros( priors.shape[0] ) 
			for i in range(len(priors_type)):
				if priors_type[i] == "gaussian":
					start[i] = priors[i,0] 		# start at peak of prior
				else:
					start[i] = ( priors[i].min() + priors[i].max() ) /2.
				print( "| %s | %s | %s | %s |" %(self.nLabels()[i], priors_type[i], priors[i], start[i]) )

		return priors, priors_type, start


	# ------------------------------------------- COMBINATION INFO ------------------------------------------------------
	def Filter_Combine_Info(self, stat_num):
		return self.pinput_combs.split('COMBINATION %s ' %stat_num)[-1].split('COMBINATION')[0]


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

	def PlotCombinedLS(self, stat_num):
		try:
			return eval(self.Filter_Combine_Info(stat_num).split('Linestyle = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except SyntaxError:
			return '-'
        
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

	def LoadPred(self, pinput, stat_num):
		nbins = self.nBins(stat_num)
		bins_to_use = self.Bins_To_Use(stat_num)
		predfile = self.PredFile(pinput, stat_num)

		if predfile[-4:] == ".npy":
			# Read in pickled file
			y_precut = np.load( predfile )
			# apply cuts 
			y = y_precut[:,bins_to_use]
			x = None # no x-predictions read in
				
		else:
			predIDs = self.PredIDs(pinput, stat_num)
			predcols = self.PredCols(pinput, stat_num)

			y = np.zeros([ len(predIDs), len(bins_to_use) ])
			for i in range( len(predIDs) ):
				pf = predfile.replace('XXXX', str(predIDs[i]))
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
			print("Applying Hartlap scaling to covariance.")
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
	


	def LoadPredNodes(self, pinput, stat_num, pred_OR_trial):
		if "pred" in pred_OR_trial:
			nodesfile = self.PredNodesFile(pinput, stat_num, 'PredNodesFile' )
			cols = self.PredNodesCols(pinput, stat_num, 'PredNodesCols' )
		elif "trial" in pred_OR_trial:
			# only will read in Trial Nodes when doing a 1D or 2D likelihood line/grid.
			nodesfile = self.PredNodesFile(pinput, stat_num, 'TrialNodesFile' )
			cols = self.PredNodesCols(pinput, stat_num, 'TrialNodesCols' )

		nodes = np.loadtxt(nodesfile, usecols=(cols))
		# if it's 1D, reshape it:
		if len(nodes.shape)==1:
			nodes = nodes.reshape(-1,1)
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
	
		nodesfile = self.PredNodesFile(self.pinput_stats, stat_num, 'TrialNodesFile' )
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
				pred = self.LoadPred(self.pinput_stats, stat)[1]
			else:
				pred = np.concatenate( (pred,self.LoadPred(self.pinput_stats, stat)[1]), axis=1 )
		return pred


	def Implement_S8Prior(self, LogL, stat_num):
		S8_Bounds = self.S8_Bounds()
		#print("Applying an S8 prior to the likelihood with bounds: ", S8_Bounds)

		if "Emu" in self.OneD_TwoD_Or_nD():
			# If emulating a 2D grid, get the x,y coords from the trial nodes file
			xc, yc = self.LoadNodes4Plot(stat_num)
		else:
			# but if reading in pre-made predictions, get x,y from PredNodesFile
			nodes = self.LoadPredNodes(self.pinput_stats, stat_num, 'pred') 
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
			nodes = self.LoadPredNodes(self.pinput_stats, stat_num, 'pred') 
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
					print( "Hit the wall searching for the %s prob-fraction on dimension %s. " %(fraction,axis))
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
	def lnprior(self, p):
		# scroll through statistics, load priors, see if point is outside allowed range.
		if self.Priors_Type[0] == "corr-gaussian":
			# a cov across the sampled params has been provided; use this as the cov of gauss prior:
			mean = np.array([ 0.2905, 0.8360, 0.6898, -1.000 ]) # THIS IS TEMP: SHOULD BE READ IN!
			mvn = multivariate_normal(mean=mean, cov=self.Priors)
			return ( mvn.logpdf(p) - mvn.logpdf(mvn.mean) )

		tot_lnprior = 0. 
		for i in range(len(p)):
			if self.Priors_Type[i] == "uniform":
				if (p[i] < self.Priors[i,0]) or (p[i] > self.Priors[i,1]): return -np.inf # kill proposal (beyond bounds)
				# no else statement needed (would only add 0)

			elif self.Priors_Type[i] == "gaussian":
				# sample from gaussian:
				mvn = multivariate_normal(mean=self.Priors[i,0], cov=self.Priors[i,1])
				tot_lnprior += ( mvn.logpdf(p[i]) - mvn.logpdf(mvn.mean) )
				# 2nd term rm normalis'n (since lhd is unnorm'd as well)
		return tot_lnprior

	def Apply_Norm(self, pred):
		# rescale training points to mean 0 and stdev 1:
		scaler = preprocessing.StandardScaler().fit( pred )
		scaled_pred = scaler.transform( pred )  
		return scaler, scaled_pred


	# if a transformation (scaling/log/PCA is called for, apply it here)
	# Note BFs & Pred_Mean optional input here (in case want to do PCA with pre-defined BFs), but...
	# ...nowhere in this script is this func utilised (FYI) so might be unnecessary.
	def Apply_Transformation(self, pinput, stat, Train_Pred, Train_BFs=None, Train_Pred_Mean=None, Train_x=None):
		scaler = None # gets redefined if Transform=="Norm" and we're using sklearn preprocessing func.

		# Identify the transformation for this statistic
		if self.Transform(pinput,stat) == "log":  
			Train_Pred = np.log( Train_Pred )
		elif "xy" in self.Transform(pinput,stat):
			# multiply y by x (e.g. making theta*xi in case of 2PCF)
			try:
				scale = float( self.Transform(pinput,stat).split('xy')[-1] )
			except ValueError:
				scale = 1.
			Train_Pred *= (Train_x*scale)
		elif self.Transform(pinput,stat) == "Norm":
			# perform the normalis'n per xbin, if the GP model is per xbin:
			if self.Model(pinput, stat) == "GPperbin":
				scaler = []
				for t in range(len(Train_x)):
					tmp_scaler, tmp_pred = self.Apply_Norm(Train_Pred[:,t].reshape(-1,1))
					Train_Pred[:,t] = tmp_pred.flatten()
					scaler.append(tmp_scaler)
			else:
				scaler, Train_Pred = self.Apply_Norm(Train_Pred)

		elif "None" in self.Transform(pinput,stat) or '------' in self.Transform(pinput,stat):
			print("Performing no transform on the input training data.")
		else:
			print( "Only log and xyN (x-TIMES-y-TIMES-N) transforms are supported. Not %s. EXITING." %self.Transform(pinput,stat) )
			sys.exit()

		if self.Perform_PCA(pinput,stat):
			PCAC = PCA_Class(self.n_components(stat))
			# either do PCA on Train_Pred directly, or using pre-defined BFs/Mean
			if type(Train_BFs) == type(None):
				Train_BFs, Train_Weights, Train_Recons = PCAC.PCA_BySKL(Train_Pred)
				Train_Pred_Mean = np.mean( Train_Pred, axis=0 )
			else:
				Train_Weights,_ = PCAC.PCA_ByHand(self, Train_BFs, Train_Pred, Train_Pred_Mean)
			inTrain_Pred = np.copy( Train_Weights )

		else:
			inTrain_Pred = np.copy(Train_Pred)
			Train_BFs = []        # dummy to be stored in case you have a mix of stats which do/dont use PCA in this combination.				
			Train_Pred_Mean = []  # ^same. Need these to preserve order of stored BFs and Train Pred Means.

		return inTrain_Pred, Train_BFs, Train_Pred_Mean, scaler 

	# remove any transformation (scaling/log/PCA) from some predictions
	def Remove_Transformation(self, pinput, stat, Pred, Train_BFs=None, Train_Pred_Mean=None, Train_x=None, scaler=None):
		# Un-do the PCA if appropriate
		if self.Perform_PCA(pinput,stat):
			PCAC = PCA_Class(self.n_components(stat))
			out_Pred = PCAC.Convert_PCAWeights_2_Predictions(Pred, Train_BFs, Train_Pred_Mean)
		else:
			out_Pred = Pred

		# Un-do the transformation:
		if self.Transform(pinput,stat) == "log":  
			out_Pred = np.exp( out_Pred )
		elif "xy" in self.Transform(pinput,stat):
			try:
				scale = float( self.Transform(pinput,stat).split('xy')[-1] )
			except ValueError:
				scale = 1.
			out_Pred = out_Pred/(Train_x*scale)
		elif self.Transform(pinput,stat) == "Norm":
			# the normalis'n is per xbin (so diff scaler), if the GP model is per xbin:
			if self.Model(pinput, stat) == "GPperbin":
				for t in range(len(Train_x)): out_Pred[t] = scaler[t].inverse_transform( out_Pred[t].reshape(-1,1) ).flatten()
			else:
				out_Pred = scaler.inverse_transform( out_Pred.reshape(1,-1) ).flatten()
		return out_Pred 


	def linear(self, x0): 
		# linear model used to fit sys-bias per bin (e.g. per theta for 2pcf) as a func of Sys_Nodes x (e.g. dz).
		# here x0 is the val of Sys_Nodes where bias=0. IMPLEMENTATION BELOW ASSUMES bias=0 @ SYS_PARAM=0;
		# made this the case for Baryons & dz, but could be false for other sys in general. 
		# Because curve_fit doesnt play nice with fixed params (x0), need an outer & an inner loop to provide this.
		def inner_linear(x, m):
			return m*(x-x0)	
		return inner_linear

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
		Train_scaler_store    = [] # if Transform is Norm (using sklearn preproc'ing), store scaler func'n

		# same, but to store info relating to systematics training set:
		Sys_HPs_store       = []
		inSys_Pred_store    = [] 
		Sys_Pred_Mean_store = []
		Sys_BFs_store       = []
		Sys_scaler_store    = []
		Sys_cov_store	    = [] # if there's corr'd nuis params, store the cov describing this corr
					 # or more specifically its cholesky transform (saves time in Lhd eval.)
		if type(self.pfile_sys) != type(None):
			for ps in range(len(self.pinput_sys)): 	# cycle through sys-files
				Sys_HPs_store.append([])
				inSys_Pred_store.append([])
				Sys_Pred_Mean_store.append([])
				Sys_BFs_store.append([])
				Sys_scaler_store.append([])
				Sys_cov_store.append([])
		# Now sys storage vectors have right dimensionality


		for stat in comb:
			print("----------------------------- ON STAT %s IN COMB %s -----------------------------" %(stat, comb), flush=True)
			# Load cosmological training set
			Train_x, Train_Pred = self.LoadPred(self.pinput_stats, stat)	
			# Apply transformation (if specified)	
			inTrain_Pred, Train_BFs, Train_Pred_Mean, Train_scaler = self.Apply_Transformation(self.pinput_stats, stat, Train_Pred, Train_x=Train_x)
				
			# storing info for each stat in the combination:
			Train_x_store.append( Train_x )
			inTrain_Pred_store.append( inTrain_Pred )
			Train_BFs_store.append( Train_BFs )
			Train_Pred_Mean_store.append( Train_Pred_Mean )
			Train_scaler_store.append( Train_scaler )

			# Now load systematics training set (if specified):
			if type(self.pfile_sys) != type(None):
				for ps in range(len(self.pinput_sys)):  # cycle through sys-files (IA,dz,BaryONs etc.)
					Sys_x, Sys_Pred = self.LoadPred(self.pinput_sys[ps], stat)
					# apply transformation (note, the binning of Train & Sys pred should be the same,
					# i.e. Train_x & Sys_x should be identical or we'll get apples & oranges):
					inSys_Pred, Sys_BFs, Sys_Pred_Mean, Sys_scaler = self.Apply_Transformation(self.pinput_sys[ps], stat, Sys_Pred, Train_x=Train_x) 
					# store sys:
					inSys_Pred_store[ps].append( inSys_Pred )
					Sys_BFs_store[ps].append( Sys_BFs )
					Sys_Pred_Mean_store[ps].append( Sys_Pred_Mean )
					Sys_scaler_store[ps].append( Sys_scaler )

			if self.OneD_TwoD_Or_nD() == "nD":
				# Train GP emulator on predictions
				print( "Running the cosmol emulator once with 1000 restarts to get the HPs." )
				Train_Nodes = self.LoadPredNodes(self.pinput_stats, stat, 'pred')
				GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Train_Nodes)	
				_,_,HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, 1000 )	
				print(HPs, flush=True)	
				HPs_store.append( HPs )

				# Now read in Sys & decide on modelling
				if type(self.pfile_sys) != type(None):
					for ps in range(len(self.pinput_sys)):  # cycle through sys-files
						Sys_Nodes = self.LoadPredNodes(self.pinput_sys[ps], stat, 'pred')

						# see if there's multiple (corrl'd) nuis params & if so, store (cholesky transf) of cov matrix
						Np_sys = self.Nparams_sys(self.pinput_sys[ps])
						if Np_sys > 1:
							sys_cov = self.SysCov(self.pinput_sys[ps])
							Sys_cov_store[ps].append( np.linalg.cholesky(sys_cov) )
						else:
							Sys_cov_store[ps].append( None )
						
						# Determine whether we are GP-emulating, or simple linear fit per bin:
						if self.Model(self.pinput_sys[ps], stat) == "Linear":
							# simple linear fit per bin of the stat (per theta in 2PCF example)
							print("Using a linear model fit for this sys-bias. HPs (gradients per bin) are...:")
							Sys_HPs = []
							for t in range(len(Train_x)):
								# here we set x0=0 (place where bias=0); also access most recent stat read in (hence [-1]):
								h,_ = curve_fit(self.linear(x0=0), Sys_Nodes.flatten(), inSys_Pred_store[ps][-1][:,t])
								Sys_HPs.append(h[0])
							print(Sys_HPs, flush=True)
							Sys_HPs_store[ps].append(Sys_HPs) # store lin fit grads for all theta bins

						elif self.Model(self.pinput_sys[ps], stat) == "GPperbin":
							which_params = self.which_params(self.pinput_sys[ps], stat)
							print("Using a GPemu PER xbin (which will take as input nusiance params: %s)" %which_params)
							print("Here's the HPs for the %s xbins:" %len(Train_x))
							Sys_HPs = [] 
							for t in range(len(Train_x)):
								inpred = inSys_Pred_store[ps][-1][:,t].reshape(-1,1)
								GPR_Class_Sys = GPR_Emu( Sys_Nodes, inpred, np.zeros_like(inpred), Sys_Nodes)
								_,_,h = GPR_Class_Sys.GPRsk(np.zeros(Sys_Nodes.shape[1]+1), None, 150 )
								Sys_HPs.append(h)
								print("%s....: "%(t+1), h, flush=True)
							Sys_HPs_store[ps].append(Sys_HPs)

						else:
							# GP emu it is. 
							print( "Running the sys emulator once with 1000 restarts to get the HPs." )
							GPR_Class_Sys = GPR_Emu( Sys_Nodes, inSys_Pred_store[ps][-1], 
								np.zeros_like(inSys_Pred_store[ps][-1]), Sys_Nodes)
							_,_,Sys_HPs = GPR_Class_Sys.GPRsk(np.zeros(Sys_Nodes.shape[1]+1), None, 1000 )
							print(Sys_HPs, flush=True)
							Sys_HPs_store[ps].append(Sys_HPs)


			# Note Giblin: 2D/1D Lhd is has no systematics functionality (since dim>2 if we include sys).
			elif self.OneD_TwoD_Or_nD() == "1DEmu" or self.OneD_TwoD_Or_nD() == "2DEmu":  
				# It's a 2D grid or 1D line of lhood evaluations & we are 
				# emulating all trial predictions here:
				Train_Nodes = self.LoadPredNodes(self.pinput_stats, stat, 'pred')
				Trial_Nodes = self.LoadPredNodes(self.pinput_stats, stat, 'trial')
				GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Trial_Nodes )	
				GP_AVOUT, GP_STDOUT, GP_HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, self.n_restarts_optimizer(stat) )	
				
				# Remove transformation from emul'd stats:
				GP_Pred = self.Remove_Transformation(self.pinput_stats, stat, GP_AVOUT,
					Train_BFs=Train_BFs, Train_Pred_Mean=Train_Pred_Mean, Train_x=Train_x)

				# concatenate the combined statistics
				if stat==comb[0]:
					Trial_Pred_store = GP_Pred
				else:
					Trial_Pred_store = np.concatenate( (Trial_Pred_store, GP_Pred), axis=1 )

		# store everything to be used by the lnlike function
		self.HPs_4thiscomb 			= HPs_store
		self.Train_x_4thiscomb 			= Train_x_store	 
		self.inTrain_Pred_4thiscomb 		= inTrain_Pred_store 	 
		# --- Following only get used if Perform_PCA is true ---
		self.Train_BFs_4thiscomb 		= Train_BFs_store  		 
		self.inTrain_Pred_Mean_4thiscomb	= Train_Pred_Mean_store 
		# --- Only gets used if Transform is set to Norm ---
		self.Train_scaler_4thiscomb 		= Train_scaler_store

		# same but for the systematics:
		self.Sys_HPs_4thiscomb		= Sys_HPs_store 
		self.inSys_Pred_4thiscomb 	= inSys_Pred_store 
		self.Sys_BFs_4thiscomb		= Sys_BFs_store
		self.inSys_Pred_Mean_4thiscomb 	= Sys_Pred_Mean_store   
		self.Sys_scaler_4thiscomb	= Sys_scaler_store
		self.Sys_cov_4thiscomb		= Sys_cov_store

		if self.OneD_TwoD_Or_nD() == "1DEmu" or self.OneD_TwoD_Or_nD() == "2DEmu":   
			return Trial_Pred_store

		return


	# log likelihood (executing emulator at each step). 
	def lnlike(self, p, cov, data, comb):

		# scroll through the statistics in this combination, 
		# read in the predictions (emulator needs these even when trained),
		# and emulate prediction for this cosmology.
		# start by checking the TrainPred has been correctly set for this combo of stats:
		if type(self.inTrain_Pred_4thiscomb) == type(None):
			self.Assemble_TrainPred_and_HPs( comb )

		# if including systematics,split the proposed coord into cosmol. & sys. params:
		# use dimensionality of HPs to make the split.
		if type(self.pfile_sys) != type(None):
			Ndim_cos = len(self.HPs_4thiscomb[0])-1 # (-1 is for the amplitude HP)
			p_cos = p[:Ndim_cos] 			# cosmol params
			p_sys = p[Ndim_cos:]			# sys params
		else:
			p_cos = p 


		GP_Pred_All = np.ones([]) # Store the emulated predictions for each stat used in the combination.
		count = 0                 # count increments through statistics
		for stat in comb:
			# Grab the important data from memory needed for the likelihood evaluation
			HPs 		= self.HPs_4thiscomb[count]
			Train_x 	= self.Train_x_4thiscomb[count]	 
			inTrain_Pred 	= self.inTrain_Pred_4thiscomb[count]       
			Train_BFs 	= self.Train_BFs_4thiscomb[count]          		 
			Train_Pred_Mean = self.inTrain_Pred_Mean_4thiscomb[count]
			Train_scaler 	= self.Train_scaler_4thiscomb[count] # only used if Transform==Norm for a stat

			# Run the (comsological) emulator
			Train_Nodes = self.LoadPredNodes(self.pinput_stats, stat, 'pred')		
			GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), p_cos.reshape(1,-1) )	
			GP_AVOUT,_,_ = GPR_Class.GPRsk(HPs, None, 0 )	

			# Remove transformation from emul'd stats:
			# Output is shaped (1,n_components) --> need to select [0,:]
			GP_Pred = self.Remove_Transformation(self.pinput_stats, stat, GP_AVOUT[0,:],
							Train_BFs=Train_BFs, Train_Pred_Mean=Train_Pred_Mean, Train_x=Train_x,
							scaler=Train_scaler)

			# Emulate the systematic contribution: 
			if type(self.pfile_sys) != type(None):

				# sum the nuisance params per stat:
				Np_sum = 0 # this is required to pull out the right sys params from p_sys
				for ps in range(len(self.pinput_sys)):  # cycle through sys-files

					# how many nuisance params associated with this systematic?
					Np_sys = self.Nparams_sys(self.pinput_sys[ps])
					# narrow p_sys to just those associated with this systematic:
					p_sys_select = p_sys[Np_sum : Np_sum+Np_sys]
					Np_sum += Np_sys

					# now we've found which BLOCK of sys params to use, identify the specific ones for this stat;
					# AND apply a cov_mat which injects the corr between nuisance params:
					if Np_sys>1:						
						L = self.Sys_cov_4thiscomb[ps][count]
						p_sys_select = L @ np.array(p_sys_select) # now they are corrl'd sampled params

						# multiple nuis params for this sys; which ones to use for this stat?
						which_params = self.which_params(self.pinput_sys[ps], stat)
						p_tmp = []
						for wp in which_params: p_tmp.append( p_sys_select[wp] )
						p_sys_select = np.array( p_tmp )
					p_sys_select = p_sys_select.reshape(1,-1)
					
					# Important stored data for sys prediction:
					Sys_HPs 	= self.Sys_HPs_4thiscomb[ps][count]
					inSys_Pred 	= self.inSys_Pred_4thiscomb[ps][count]       
					Sys_BFs 	= self.Sys_BFs_4thiscomb[ps][count]          		 
					Sys_Pred_Mean 	= self.inSys_Pred_Mean_4thiscomb[ps][count]
					Sys_Nodes 	= self.LoadPredNodes(self.pinput_sys[ps], stat, 'pred')
					Sys_scaler 	= self.Sys_scaler_4thiscomb[ps][count]

					# here implement which_params functionality
					if self.Model(self.pinput_sys[ps], stat) == "Linear":
						# do linear model per x-bin (theta in 2PCF case)
						Sys_AVOUT = np.zeros([ 1,len(Train_x) ])
						for t in range(len(Train_x)):
							# Again we assume x0 (place where bias=0) is 0
							Sys_AVOUT[0,t] = self.linear(x0=0)(p_sys_select, Sys_HPs[t])

					elif self.Model(self.pinput_sys[ps], stat) == "GPperbin":
						# do a GPemu PER xbin, poss using multiple nuisance params....
						Sys_AVOUT = np.zeros([ 1,len(Train_x) ])
						for t in range(len(Train_x)):
							inpred = inSys_Pred[:,t].reshape(-1,1)
							GPR_Class_Sys = GPR_Emu( Sys_Nodes, inpred, np.zeros_like(inpred), p_sys_select )
							Sys_AVOUT[0,t],_,_ = GPR_Class_Sys.GPRsk(Sys_HPs[t], None, 0 )

					else:
						# GP-emu it is. 
						GPR_Class_Sys = GPR_Emu( Sys_Nodes, inSys_Pred, np.zeros_like(inSys_Pred), p_sys_select )
						Sys_AVOUT,_,_ = GPR_Class_Sys.GPRsk(Sys_HPs, None, 0 )
					# rm transformation:
					Sys_Pred = self.Remove_Transformation(self.pinput_sys[ps], stat, Sys_AVOUT[0,:],
								Train_BFs=Sys_BFs, Train_Pred_Mean=Sys_Pred_Mean, Train_x=Train_x, # (Train_x=Sys_x)
								scaler=Sys_scaler)
					GP_Pred+=Sys_Pred # add sys contribution

			# Combine the predictions
			GP_Pred_All = np.append( GP_Pred_All, GP_Pred ) 
			
			# increment statistics count.
			count +=1

		GP_Pred_All = np.delete( GP_Pred_All, 0 )  # get rid of first element (comes from initialisation) 
		LnLike = -0.5 * np.dot( np.transpose(data - GP_Pred_All), np.dot(np.linalg.inv(cov), (data - GP_Pred_All)  ))
		return LnLike #, GP_Pred_All

	# log posterior
	def lnprob(self, p, cov, data, comb):
		lp = self.lnprior(p)
		return lp + self.lnlike(p, cov, data, comb) if np.isfinite(lp) else -np.inf

	def Run_MCMC(self, comb_num, comb):

		#import emcee
		from scipy.stats import norm
		from nautilus import Prior
		from nautilus import Sampler
		import time

		# Load the data vector and covariance to be used in this sampling
		cov = self.LoadCovCombined(comb_num)
		data = self.CombineData( comb )
		
		# Load priors (bounds VS gauss mean/stdev), prior types (uniform VS gauss) & starting position
		self.Priors, self.Priors_Type, p = self.Priors_Start_MCMC(comb[0]) 
		ndim = len(p)
		# assemble nautilus prior object
		prior = Prior()
		for i in range(ndim):
			if self.Priors_Type[i] == "uniform":
				prior.add_parameter( self.nLabels()[i], dist=(self.Priors[i,0],self.Priors[i,1] ))
			elif self.Priors_Type[i] == "gaussian":
				prior.add_parameter( self.nLabels()[i], dist=norm(self.Priors[i,0],self.Priors[i,1]))
			else:
				print( "Only uniform and gaussian priors are supported. Not %s. EXITING." %self.Priors_Type[i] )
				sys.exit()

		def naut_likelihood(theta):
			# Nautilus may pass a dict (name->value) or an array-like.
			if isinstance(theta, dict):
				# use the same ordering as prior construction (self.nLabels())
				p_arr = np.array([theta[name] for name in self.nLabels()[:ndim]])
			else:
				p_arr = np.array(theta)
			# return lnlike (not lnprob) so prior handled by Nautilus
			return float(self.lnlike(p_arr, cov, data, comb))

		# get name for checkpoint file:
		checkname = "%s/Checkpoint_Samples_SurveySize%s_%s" %(self.savedirectory(), 
														self.SurveyCombinedArea(comb_num), self.CombName(comb_num) )
		sampler = Sampler(prior, naut_likelihood, n_live=1000, filepath='%s.hdf5'%checkname)
		t_start = time.time()
		sampler.run(verbose=True)
		t_end = time.time()
		print('Nautilus run time: {:.1f} minutes'.format((t_end - t_start)/60.))
		samples, log_w, log_l = sampler.posterior()

		# hopefully don't need the following if we've got
		# the version of nautilus correct.
		"""
		# Attempt a few common ways to extract posterior samples from Nautilus.
		# Depending on Nautilus version use get_samples(), posterior, samples, or similar.
		try:
			samples = np.asarray(sampler.get_samples())
		except Exception:
			try:
				samples = np.asarray(sampler.posterior)   # some APIs expose .posterior
			except Exception:
				try:
					samples = np.asarray(sampler.samples)   # fallback attribute
				except Exception:
					raise RuntimeError("Unable to extract posterior samples from Nautilus sampler object. Inspect Nautilus API.")
		"""

		# old script using emcee below:
		""" 
		#ndim = len(p)
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
		"""

		# Build a result dict so caller can access raw nautilus outputs too
		try:
			result = {
				'samples': samples,
				'log_weights': log_w,
				'log_likelihoods': log_l
			}
		except NameError:
			# fallback if those names don't exist (e.g. emcee path)
			result = {
				'samples': samples,
				'log_weights': None,
				'log_likelihoods': None
			}

		# Just in case the parameter file specifies MULTIPLE combinations of stats, we need to RESET
		# the following variables, which stored data specific to each combination of statistics in memory.
		self.Train_x_4thiscomb 		= None		 
		self.inTrain_Pred_4thiscomb 	= None   	 
		self.Train_BFs_4thiscomb 	= None		 
		self.inTrain_Pred_Mean_4thiscomb = None      
		self.HPs_4thiscomb 		= None    
		# same for sys:
		if type(self.pfile_sys) != type(None):
			self.inSys_Pred_4thiscomb 	= None   	 
			self.Sys_BFs_4thiscomb 		= None		 
			self.inSys_Pred_Mean_4thiscomb 	= None      
			self.Sys_HPs_4thiscomb 		= None  
		
		return result



	def Master_Run_MCMC(self, comb_num, comb):
		print( "---------------- RUNNING STATS COMBO %s "%comb_num, comb, "---------------------------------------------- " )
		if not os.path.exists(self.savedirectory()):
	 		os.makedirs(self.savedirectory())

		savename = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_%s" %(self.savedirectory(), 
			self.SurveyCombinedArea(comb_num), self.nwalkers(), self.real_steps(), self.CombName(comb_num) )
		samples = self.Run_MCMC(comb_num, comb)
		#np.save( savename, samples )
		#self.Plot_MCMC_Lhd(samples, savename)
		
		# Run_MCMC may return a dict (when using Nautilus) or a plain samples array (emcee)
		if isinstance(samples, dict):
			# save legacy plain samples file for downstream code expecting .npy
			np.save(savename, samples['samples'])
			# also save the full output (including points, weights, log-likes)
			np.savez(savename + '_full.npz',
			         samples=samples['samples'],
			         points=samples.get('points'),
			         log_weights=samples.get('log_weights'),
			         log_likelihoods=samples.get('log_likelihoods'))
			# for returning keep the plain samples array for backward compat
			samples_out = samples['samples']
		else:
			np.save(savename, samples)
			samples_out = samples

		print( "---------------- FINISHED STATS COMBO %s "%comb_num, comb, "---------------------------------------------- " )
		return samples_out, savename



	# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
	def Plot_2D_Lhd(self, Log_Lhds_Stats, Contours_Stats, 
							  Log_Lhds_Comb, Contours_Comb):

		font = {'family' : 'serif',
                'weight' : 'normal',
                        'size'   : 20}                                                                                                         
		plt.rc('font', **font)

                # Plot a 2D likelihood grid
		Use_Stats = self.Use_Stats()
		LW = 2 # Linewidths
		handles = []
		
		plt.figure(figsize=(11,9))

		# scroll through the statistics to plot contours for
		for i in range( len(Use_Stats) ):

			if "Emu" in self.OneD_TwoD_Or_nD():
				# If emulating a 2D grid, get the x,y coords from the trial nodes file
				xc, yc = self.LoadNodes4Plot(Use_Stats[i])
			else:
				# but if reading in pre-made predictions, get x,y from PredNodesFile
				nodes = self.LoadPredNodes(self.pinput_stats, Use_Stats[i], 'pred') 
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
				nodes = self.LoadPredNodes(self.pinput_stats, Combine_Stats[i][0], 'pred') 
				xc = nodes[:,0]
				yc = nodes[:,1]

			
			plt.contour(Log_Lhds_Comb[i], [Contours_Comb[i][0], Contours_Comb[i][1]], 
						origin='lower', extent = [xc.min(), xc.max(), yc.min(), yc.max()], linewidths=LW, 
						colors=self.PlotCombinedColour(i+1), linestyles=self.PlotCombinedLS(i+1) )
			handles.append( mlines.Line2D([],[],color=self.PlotCombinedColour(i+1), linestyle=self.PlotCombinedLS(i+1),
                                                      linewidth=LW, label=self.PlotCombinedLabel(i+1)) ) 
		
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
	                'size'   : 48}                                                                                 
		plt.rc('font', **font)
		lw = 6
		max_n_ticks = 3

		# Scroll through combinations of statistics, read in samples & plot their contours.
		constraints = [] # save and return the mean & asymmetric 68% confidence regions.
		handles = []
	
		# If plot_limits set in parameter file, read them in:
		try:
			limits = eval( self.pinput_combs.split('Plot_Limits = ')[-1].split('#')[0].split('\n')[0] )
		except SyntaxError:
			# If no limits were read in, set them to the range of the first set of samples.
			comb_num = 1
			sample_name = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_%s.npy" %(self.savedirectory(),self.SurveyCombinedArea(comb_num),self.nwalkers(), self.real_steps(), self.CombName(comb_num) )
			samples = np.load( sample_name )
			limits = []
			for j in range(samples.shape[1]):
				limits.append([ samples[:,j].min(), samples[:,j].max() ]) 

		print("The plot limits have been set to: ", limits) 

		# Decide what dimensions are getting plotted (if Plot_Dims not set, all will be plotted)
		# But note, if Plot_Dims NOT set, you'll get problems if you try to plot chains of different dimensions:
		PD = self.Plot_Dims()
		if type(PD) != type(None): 
			# Plot_Dims has been specified
			print("Just plotting the following dimensions of the samples: ", PD)
			plot_labels = []
			plot_lims = []
			for d in range(len(PD)):
				plot_labels.append(self.nLabels()[PD[d]])
				plot_lims.append(limits[PD[d]])
		else: 
			# if Plot_Dims not specified, plot all dimensions
			plot_labels = self.nLabels()
			plot_lims = limits
			PD = range(len(plot_labels))

		# Check if a truth file exists.
		import os
		if os.path.exists(self.DataNodesFile()):			
			truth = self.LoadDataNodes()[PD]
		else:
			print("Not plotting truth point since no DataNodesFile found.") 
			truth = np.zeros( len(PD) ) + np.nan

		for i in range( len(self.Combine_Stats()) ):
			comb_num = i+1
			# sample_name matches the one used in Master_Run_MCMC
			sample_name = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_%s.npy" %(self.savedirectory(),self.SurveyCombinedArea(comb_num),self.nwalkers(), self.real_steps(), self.CombName(comb_num) )
			samples = np.load( sample_name )
			sample_name_full = sample_name.replace('.npy','_full.npz')
			if os.path.exists(sample_name_full):
				# if the full nautilus output file exists, read weights from there
				data = np.load(sample_name_full)
				log_w = data['log_weights']
				weights = np.exp(log_w - np.max(log_w))  # subtract max for numerical stability
				print("Will apply Nautilus weights to the samples for plotting.")
				# dont need to downsample here, since weights applied in corner function
				# checked and it gives the same answer.
				#samples = samples[np.random.choice(samples.shape[0], size=100000, p=weights/np.sum(weights)), :]
			else:
				weights = None
			# If working with sys, some of the sys-params could be corr'd, but the ones saved to the chain are UNcorr'd,
			# so we need to apply the corr'n rot'n here to the relevant columns:
			# NOTE: if statement is used in case of plotting Sys-free & Sys @ same time, and...
			# ... it ASSUMES 4 COSMOL DIMENSIONS.
			Ndim_cos = 4
			if type(self.pfile_sys) != type(None) and samples.shape[1]>Ndim_cos:  
				# sum the nuisance params per stat:
				Np_sum = Ndim_cos # this is required to pull out the right sys params from p_sys
				for ps in range(len(self.pinput_sys)):  # cycle through sys-files

					# how many nuisance params associated with this systematic?
					Np_sys = self.Nparams_sys(self.pinput_sys[ps])
					# if more than 1 (corr'd) sys params, apply the rot'n to the relevant cols:
					if Np_sys>1:
						sys_cov = self.SysCov(self.pinput_sys[ps])
						L = np.linalg.cholesky(sys_cov)
						for step in range(samples.shape[0]): # rotate every step in the chain
							samples[step, Np_sum : Np_sum+Np_sys] = L @ samples[step, Np_sum : Np_sum+Np_sys] 						
					Np_sum += Np_sys

			# Narrow to just the plotting range:
			samples = samples[:,PD] 

			if self.SmoothCombinedContour(comb_num):
				SS = self.SmoothCombinedScale(comb_num)
				print("Read in contour smoothing scale of %s pxls" %SS)
			else:
				SS=1

			if i==0:
				# Then it's the first set of samples, establish fig:
				fig = corner.corner(samples, weights=weights, labels=plot_labels, range=plot_lims,
						plot_contours=True, plot_density=False, plot_datapoints=False, smooth=SS,
						levels=(0.68,0.95), truths=truth, truth_color='black', 
						contour_kwargs={'colors':[self.PlotCombinedColour( comb_num )], 'linewidths':lw, 
										'linestyles':[self.PlotCombinedLS( comb_num )]},
						hist_kwargs={'color':[self.PlotCombinedColour( comb_num )], 'linewidth':lw},
						max_n_ticks=max_n_ticks)

			else:
				# sequential set of contours - overplot them
				fig = corner.corner(samples, weights=weights, labels=plot_labels, range=plot_lims,
						plot_contours=True, plot_density=False, plot_datapoints=False, smooth=SS,
						levels=(0.68,0.95), truths=truth,
						contour_kwargs={'colors':[self.PlotCombinedColour( comb_num )], 'linewidths':lw,
										'linestyles':[self.PlotCombinedLS( comb_num )]},
						hist_kwargs={'color':[self.PlotCombinedColour( comb_num )], 'linewidth':lw},
						max_n_ticks=max_n_ticks, fig=fig)

			handles.append( mlines.Line2D([], [], color=self.PlotCombinedColour( comb_num ),
                                                      linestyle=self.PlotCombinedLS( comb_num ),
                                                      label=self.PlotCombinedLabel( comb_num )) )

			# extract and save the constraints:
			tmp_constraints = np.empty([ samples.shape[1], 3 ])
			for j in range( samples.shape[1] ):
				mean = corner.quantile( samples[:,j], [0.5] )[0]
				upper = corner.quantile( samples[:,j], [0.84] )[0] - mean
				lower = mean - corner.quantile( samples[:,j], [0.16] )[0] 
				tmp_constraints[j] = np.array([ mean, upper, lower ])
			constraints.append( tmp_constraints )
		
		fig.set_size_inches((25,25))
		if samples.shape[1] > 5: # dimensionality affects best place for legend
			plt.legend(handles=handles, bbox_to_anchor=[1.2,7.5], loc='upper right')
		else:
			plt.legend(handles=handles,bbox_to_anchor=(0., 2.2, 1.0, .0), loc='upper right')

		if type(savename) == type(None):
			# If no savename given use this one as default.
			# Matches the format of the sample_name, specified above and originally in Master_Run_MCMC.
			savename = "%s/Samples_SurveySize%s_GPErrorNone_nwalkers%s_nsteps%s_AllComb_Contours.png" %(self.savedirectory(), self.SurveyCombinedArea(1), self.nwalkers(), self.real_steps() )		
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
				preds = self.LoadPred(self.pinput_stats, stat)[1] 
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

			print("SmoothContour is...", self.SmoothContour(stat))
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





