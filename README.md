# Calc_Lhd_Tool
## A one-stop code to perform MCMC or 2D-grid Bayesian parameter inference using a Gaussian process emulator in Python

This software package samples the posterior probability distributions either with a Markov Chain Monte Carlo (MCMC; for dim>2) or on a 2D grid.

The basic command to execute the software is:

```
python Calc_Lhd_MCMC.py <parameter_file>		# for an N-dimension MCMC sampling of the posterior
```

Or

```
python Calc_Lhd_2D.py <parameter_file>			# for a 2D grid-based sampling of the posterior
```

The user only needs to edit the input <parameter_file> to specify the file addresses of a data vector, covariance matrix, model predictions and corresponding input (often cosmological) parameters, and the scale cuts to apply. In the case of the 2D likelihood code, the model predictions are assumed to be defined on a grid, whereas for the MCMC code, the model predictions are used to train a Gaussian process regression emulator to interpolate across the N-dimensional space. The emulator is trained before the MCMC begins and is then executed at every point in the chain to produce a prediction. In both cases, the 2D and MCMC setups, you can specify arbitrary combinations of the statistics in your parameter file, as long as you provide a corresponding combined covariance matrix. This allows you to easily compute the combined constraints from different cosmological probes, or perform tomography in a weak lensing analysis. 

## Installation

Navigate to whether on your machine you would like to install the code and run:
```
git clone https://github.com/benjamingiblin/Calc_Lhd_Tool.git .
```

**Dependencies:** it is assumed that you have an up-to-date Anaconda distribution, which contains practically everything the code needs to run. If you encounter an error that a required python package is missing, this can easily be solved with:
```
pip install <package_name>
```


## EXAMPLE 1 - A 2D-GRID PARAMETER INFERENCE PROBLEM

The git repository contains a 2D parameter inference problem in *examples/2D_Lhd/*. This can be executed with:
```
python Calc_Lhd_2D.py examples/2D_Lhd/param_files_2D/params_flask_2D-Lhd_Ommsigma8.dat
```

This example performs a mock cosmic shear (weak lensing) analysis, constraining Omegam and sigma8 using the probability density of convergence maps (or *lensing PDFs*) as the statistic of choice. This is essentially a simplified version of the analysis presented in Figure 3 of [Giblin et al. 2023][1]

The input parameter file contains information on two statistics, STATISTIC 1 and STATISTIC 2. The first two arguments of the parameter file are:
```
Use_Stats = [1,2]
Combine_Stats = [[1,2]]
```
The first arg tells the code you would like to produce the constraints for statistic 1 and 2 separately. The second arg tells the code to also produce the constraints from their combination. If more statistics were presented in the file, one could produce additional constraints with, e.g.,:
```
Use_Stats = [1,2,3,4]		# produce separate constraints for stats 1-4
Combine_Stats = [[1,2],[3,4]]   # combine stats 1&2, and then combine 3&4.
```





[1]: https://arxiv.org/abs/2211.05708 "Giblin et al."