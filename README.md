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

To get to grips with how the code works, users are encouraged to run two examples, the READMEs for which can be found in *examples/2D_Lhd* (example 1) and *examples/MCMC* (example 2).


