# Calc_Lhd_Tool
## A one-stop code to perform MCMC or 2D-grid Bayesian parameter inference using a Gaussian process emulator in Python

## EXAMPLE 1 - A 2D-GRID PARAMETER INFERENCE PROBLEM

### Overview

The git repository contains a 2D parameter inference problem in *examples/2D_Lhd/*. This can be executed with:
```
python Calc_Lhd_2D.py examples/2D_Lhd/param_files_2D/params_flask_2D-Lhd_Ommsigma8.dat
```

This example performs a mock cosmic shear (weak lensing) analysis, constraining the cosmological parameters Omegam and sigma8 using the probability density of convergence maps (or *lensing PDFs*) as the statistic of choice. This is essentially a simplified version of the analysis presented in the right-hand panel of Figure 3 of [Giblin et al. 2023][1]. The model predictions, data, and covariance matrices were all measured from full-sky noise-free lognormal convergence maps produced using FLASK [(Xavier et al. 2017)][2].

The input parameter file contains information on two statistics, appearing under the STATISTIC 1 and STATISTIC 2 banners. Statistic 1 and 2 respectively are lensing PDFs measured from maps smoothed on the scale of 14arcmin and 60arcmin respectively. The first two arguments of the parameter file are:
```
Use_Stats = [1,2]		# Which statistics in this params_file to use in analysis
Combine_Stats = [[1,2]]		# Which statistics to combine
```
The first arg tells the code you would like to produce the constraints for statistic 1 and 2 separately (i.e. using the information under the STATISTIC 1 and STATISTIC 2 banners respectively). The second arg tells the code to also produce the constraints from their combination. If more statistics were presented in the file (under banners STATISTIC 3 and STATISTIC 4 etc.), one could produce additional constraints with, e.g.,:
```
Use_Stats = [1,2,3,4]		# produce separate constraints for stats 1-4
Combine_Stats = [[1,2],[3,4]]   # combine stats 1&2, and then combine 3&4.
```

### Individual statistics (specified by the *Use_Stats* argument)

The model predictions for both Statistic 1 and 2 are defined on an Omegam-sigma8 grid with dimensionality 200x200. The predictions are saved in numpy-pickled files, e.g., for STATISTIC 1, we have:
```
PredFile = examples/2D_Lhd/Predictions/PDF_linear_nofz1_SS14arcmin_Cosmols1-40000.npy
```
contains 40,000 predictions (one for each pixel on our grid), with the corresponding cosmological parameters appearing in:
```
PredNodesFile = examples/2D_Lhd/Predictions/Cosmologies_Omm_sigma8_h_w0.txt
```
There are 4 columns in this file, referring to (as the name suggests) Omegam, sigma8, dimensionless Hubble parameter, and dark energy eqn of state parameter respectively. h and w0 are fixed, as this is a 2D grid-based analysis. The following argument tells the code which columns to read as the x and y axes of the grid (here Omegam and sigma8 respectively):
```
PredCols = [0,1]
```

Each lesning PDF prediction is 6 elements long, hence,
```
nBins = 6
```
in the parameter file, and we can specify which bins get used in the likelihood (i.e. scale cuts) with
```
Bins_To_Use = range(6)
```
with this argument currently set to use all bins. If one were to change this (e.g. range(1,5) to exclude the 1st and last bin), then the necessary cuts are automatically made on the 6x6 covariance matrix specified by
```
CovFile = examples/2D_Lhd/Cov/Cov_PDF_linear_nofz1_SS14arcmin_CosmolData_Nreal1000.npy
```
Note, that the x-values associated with these predictions (the centre of the convergence bins that the PDF(convergence) was measured in) are also stored in the *Predictions* subdirectory, e.g.,
```
examples/2D_Lhd/Predictions/Kappa_linear_nofz1_SS14arcmin.txt	# convergence bins for statistic 1
examples/2D_Lhd/Predictions/Kappa_linear_nofz1_SS60arcmin.txt	# and for stat 2
```
but these are not used by the code, since they are the same for all predictions and the data vector. 


The true cosmology (an optional argument, as it is not always known) is contained in a file specified near the top of the parameter file:
```
DataNodesFile = examples/2D_Lhd/Data/Cosmology_Data_Omm_sigm8_h_w0.txt	# True cosmology, used to place a crosshair on output plot
```
As with the model predictions, there is an argument saying which columns to use from the *DataNodesFile*:
```
DataNodesCols = [0,1]
```

The covariance matrix, produced with 1000 FLASK realisations at the true cosmology, is specified for each statistic with:
```
CovFile = examples/2D_Lhd/Cov/Cov_PDF_linear_nofz1_SS14arcmin_CosmolData_Nreal1000.npy
```
This covariance (as with the predictions and data) was measured on *full-sky* HealPix maps [(Zonca et al. 2019)][3]. But we can scale the covariance to correspond to a desired sky area using these arguments:
```
CovArea = (4*np.pi*(180./np.pi)**2)	# The sky area [deg^2] this covariance was measured on (here full sky)
SurveyArea = 50				# Survey size you want to scale the cov to [deg^2]
```
This will scale the covariance by the ratio *(SurveyArea/CovArea)*. Note that python numpy syntax (*np.pi*) is used in specifying CovArea, but one could equivalently have just put 41253.
The Hartlap correction [(Hartlap et al. 2007)][4], used to combat the noise bias which affects the inversion of covariance matrices estimated from a finite number of realisations, is applied if
```
Apply_Hartlap = True
```
at the top of the parameter file, and
```
Nreal = 1000	# Number of realisations used in cov estimation
```
is specified for each statistic.




### Combinations of statistics (specified by the *Combine_Stats* argument)





[1]: https://arxiv.org/abs/2211.05708 "Giblin et al."
[2]: https://arxiv.org/abs/1602.08503 "Xavier et al."
[3]: https://joss.theoj.org/papers/10.21105/joss.01298 "Zonca et al."
[4]: https://arxiv.org/abs/astro-ph/0608064 "Hartlap et al."