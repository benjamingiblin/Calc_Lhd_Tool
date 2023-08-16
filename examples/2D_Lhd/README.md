# Calc_Lhd_Tool
## A one-stop code to perform MCMC or 2D-grid Bayesian parameter inference using a Gaussian process emulator in Python

## EXAMPLE 1 - A 2D-GRID PARAMETER INFERENCE PROBLEM

The git repository contains a 2D parameter inference problem in *examples/2D_Lhd/*. This can be executed with:
```
python Calc_Lhd_2D.py examples/2D_Lhd/param_files_2D/params_flask_2D-Lhd_Ommsigma8.dat
```

This example performs a mock cosmic shear (weak lensing) analysis, constraining Omegam and sigma8 using the probability density of convergence maps (or *lensing PDFs*) as the statistic of choice. This is essentially a simplified version of the analysis presented in Figure 3 of [Giblin et al. 2023][1]. The model predictions, data, and covariance matrices were all produced using FLASK [Xavier et al. 2017][2].

The input parameter file contains information on two statistics, appearing under the STATISTIC 1 and STATISTIC 2 headers. The first two arguments of the parameter file are:
```
Use_Stats = [1,2]
Combine_Stats = [[1,2]]
```
The first arg tells the code you would like to produce the constraints for statistic 1 and 2 separately. The second arg tells the code to also produce the constraints from their combination. If more statistics were presented in the file, one could produce additional constraints with, e.g.,:
```
Use_Stats = [1,2,3,4]		# produce separate constraints for stats 1-4
Combine_Stats = [[1,2],[3,4]]   # combine stats 1&2, and then combine 3&4.
```

The model predictions for both Statistic 1 and 2 are defined on an Omegam-sigma8 grid with dimensionality 200x200. The predictions are saved in numpy-pickled files, e.g., for STATISTIC 1:
```
PredFile = examples/2D_Lhd/Predictions/PDF_linear_nofz1_SS14arcmin_Cosmols1-40000.npy
```
contains 40,000 predictions (one for each pixel on our grid), with the corresponding cosmological parameters appearing in:
```
PredNodesFile = examples/2D_Lhd/Predictions/Cosmologies_Omm_sigma8_h_w0.txt
```
There are 4 columns in this file, referring to (as the name suggests) Omegam, sigma8, dimensionless Hubble parameter, and dark energy eqn of state parameter respectively. h and w0 are fixed, 


Each prediction is 6 elements long, hence,
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

The covariance was produced with 1000 FLASK realisations at the true cosmology, which is contained in a file specified near the top of the parameter file:
```
DataNodesFile = examples/2D_Lhd/Data/Cosmology_Data_Omm_sigm8_h_w0.txt
```




[1]: https://arxiv.org/abs/2211.05708 "Giblin et al."
[2]: https://arxiv.org/abs/1602.08503 "Xavier et al."