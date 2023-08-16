# Calc_Lhd_Tool
## A one-stop code to perform MCMC or 2D-grid Bayesian parameter inference using a Gaussian process emulator in Python

## EXAMPLE 1 - A 2D-GRID PARAMETER INFERENCE PROBLEM

### Overview

The git repository contains a 2D parameter inference problem in *examples/2D_Lhd/*. This can be executed with:
```
python Calc_Lhd_2D.py examples/2D_Lhd/param_files_2D/params_flask_2D-Lhd_Ommsigma8.dat
```
from inside wherever you have cloned the repository.

This example performs a mock cosmic shear (weak lensing) analysis, constraining the cosmological parameters Omegam and sigma8 using the probability density of convergence maps (or *lensing PDFs*) as the statistic of choice. This is essentially a simplified version of the analysis presented in the right-hand panel of Figure 3 of [Giblin et al. 2023][1]. The model predictions, data, and covariance matrices were all measured from full-sky noise-free lognormal convergence maps produced using FLASK [(Xavier et al. 2017)][2].

This README gives information on the most important arguments in the input parameter file.
```
examples/2D_Lhd/param_files_2D/params_flask_2D-Lhd_Ommsigma8.dat
```
**More information can be found in the comments of this parameter file.**

The output of this example is this plot of the cosmological constraints from two lensing PDFs and their combination (which are explained in more detail further down):
![Output constraints of the 2D likelihood example](https://github.com/benjamingiblin/Calc_Lhd_Tool/blob/master/examples/2D_Lhd/Results/Plot_Lhd_Omm-sigma8.png)


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

The following arguments tell the code the dimensionality of the grid-based sampling: 
```
x_Res = 200
y_Res = 200
```
The model predictions here are produced for 200x200 grid of Omegam-sigma8 values (40,000 in total). More on this in the section on **Individual statistics** below.


The true cosmology of the data is not always known, but as we are working with simulated data in this case, we can input the true vcosmology to the code, which will place a star at the relevant coordinates on the output plot. This info is given to the code near the top of the parameterfile with:
```
DataNodesFile = examples/2D_Lhd/Data/Cosmology_Data_Omm_sigm8_h_w0.txt  # True cosmology, used to place a crosshair on output plot
```
There is an argument straight after saying which columns to use from the *DataNodesFile*:
```
DataNodesCols = [0,1]
```
There are 4 columns in *DataNodesFile*, referring to (as the name suggests) Omegam, sigma8, dimensionless Hubble parameter (h), and dark energy eqn of state parameter (w0) respectively. h and w0 are fixed, as this is a 2D grid-based analysis. The first argument of *DataNodesCols* is used for the x-axis coord, and the 2nd is taken as the y-axis coord.




### Individual statistics (specified by the *Use_Stats* argument)

The model predictions for both Statistic 1 and 2 are defined on an Omegam-sigma8 grid with dimensionality 200x200. The predictions are saved in numpy-pickled files, e.g., for STATISTIC 1, we have:
```
PredFile = examples/2D_Lhd/Predictions/PDF_linear_nofz1_SS14arcmin_Cosmols1-40000.npy
```
contains 40,000 predictions (one for each pixel on our grid), with the corresponding cosmological parameters appearing in:
```
PredNodesFile = examples/2D_Lhd/Predictions/Cosmologies_Omm_sigma8_h_w0.txt
```
There are 4 columns in this file, referring to (as the name suggests) Omegam, sigma8, h, and w0, same as the *DataNodesFile*. The following argument tells the code which columns to read as the x and y axes of the grid (here Omegam and sigma8 respectively):
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


The covariance matrix, produced with 1000 FLASK realisations at the true cosmology, is specified for each statistic with:
```
CovFile = examples/2D_Lhd/Cov/Cov_PDF_linear_nofz1_SS14arcmin_CosmolData_Nreal1000.npy
```
This covariance (as with the predictions and data) was measured on *full-sky* HealPix maps [(Zonca et al. 2019)][3]. But we can scale the covariance to correspond to a desired sky area using these arguments:
```
CovArea = (4*np.pi*(180./np.pi)**2)	# The sky area [deg^2] this covariance was measured on (here full sky)
SurveyArea = 50				# Survey size you want to scale the cov to [deg^2]
```
This will scale the covariance by the ratio *(SurveyArea/CovArea)*. Note that python numpy syntax (*np.pi*) is used in specifying CovArea, but one could equivalently have just put 41253 (the area of the sky on deg2).
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

At the end of the parameter file, we have the section which defines arguments used in the various combinations of the input statistics. These appear under banners labelled *COMBINATION 1*, *COMBINATION 2*, etc.

*COMBINATION 1* refers to the first element of the *Combine_Stats* argument. Here the first element of this argument is [1,2], which means that statistics 1 and 2 are combined using a combined covariance matrix specified under the *COMBINATION 1* banner of the parameter file.
```
CovFile = examples/2D_Lhd/Cov/CovCombined_PDF_linear_nofz1_SS14arcmin-60arcmin_CosmolData_Nreal1000.npy
```

If we had more statistics present in this parameter file (under banners STATISTIC 3 and STATISTIC 4), we could add them to the *Combine_Stats* argument like so,
```
Combine_Stats = [[1,2],[3,4]]   # combine stats 1&2, and then combine 3&4.
```
and then add another combinations banner, named *COMBINATION 2*, under which the combined covariance for stats 3&4 would be specified.

The ordering of combinations in the parameter file does not matter - COMBINATION 2 could appear above COMBINATION 1. All that matters is that things under the *COMBINATION N* banner correspond to the stats specified by the N'th element of the *Combine_Stats* argument.

The other arguments of the COMBINATIONS section are hopefully self explanatory. Just as with the individual statistics, one can set arguments to scale the covariance, specify the number of realisations used in the covariance estimation (used if *Apply_Hartlap* is set to True), and specify the line colour and legend label.



### Takeaway notes (IMPORTANT, PLEASE READ)

If this example has worked for you, I encourage you to also try example 2 (*examples/MCMC*) before adapting this software package for your own purposes.

When you are ready to use this code in anger, there's some **important** things you should know before copying one of the example parameter files and editing for your own purposes:

- **Defining banners**: those '-' (dashes) either side of *STATISTIC N* and *COMBINATION N* are important, and are used by the code to filter the information pertaining to the different stats. So keep those in. The same goes for the *- END OF STATISTICS -* line (and associated dashes); leave this in as it helps the code to separate the info relating to statistcs and to combinations.

- **Spaces are important**: The spaces that appear in specifying an argument, (e.g. *DataNodesFile = <Data_Nodes_Address>*) are all important. *DataNodesFile=<Data_Nodes_Address>* (no spaces) will not be recognised, so keep those spaces people!

- **Case and spelling are important**: *DataNodesFile* is not the same as *datanodesfile* or *DataNodesFil*. The former will be recognised by the code, the latter two will not.

- **Arguments should appear only once**: The code searches for the first appearance of *<Keyword> = * in the parameter file, and it can see instances of the keyword even if they're commented out. So, to avoid any confusion, you should only have *DataNodesFile = <blah_blah>* appearing **once**, and the same goes for all other keywords. 

 - **Errors mean problems with your parameter file**: if you encounter errors, they are almost certanly down to something being mis-specified in your parameter file. This means you should search for the bug in there before you begin chaning anything under the bonnet of the code (i.e. in the *Classes_4_Lhd.py* or *Classes_4_GPR.py* scripts). Usually the error comes from a keyword being misspelt (see bulletpoint above), or the input parameter file not having the correct format.


[1]: https://arxiv.org/abs/2211.05708 "Giblin et al."
[2]: https://arxiv.org/abs/1602.08503 "Xavier et al."
[3]: https://joss.theoj.org/papers/10.21105/joss.01298 "Zonca et al."
[4]: https://arxiv.org/abs/astro-ph/0608064 "Hartlap et al."