# Calc_Lhd_Tool - a one-stop code to perform MCMC or 2D-grid likelihood evaluations using a Gaussian process emulator

This software package samples the posterior probability distributions either with a Markov Chain Monte Carlo (MCMC; for dim>2) or on a 2D grid.

The basic command to execute the software is:

```python Calc_Lhd_MCMC.py <parameter_file>		# for an N-dimension MCMC sampling of the posterior
python Calc_Lhd_2D.py <parameter_file>			# for a 2D grid-based sampling of the posterior
```

The user only needs to edit the input <parameter_file> which contains, amongst other things, the file address for a data vector, covariance matrix, and model predictions used to train a Gaussian process emulator, for 1 or more statistics. The statistics specified in the parameter file (perhaps from different redshift bins or cosmological probes) can easily be combined, or subject to scale cuts, as specified in the parameter file.

The 

runs Markov Chain Monte Carlo (MCMC) sampling of a posterior distribution, using data, covariance, and model predictions specified by the user. The model predictions are used to train a Gaussian process regression emulator before the MCMC begins. The MCMC sampler then uses the trained emulator to make a prediction at each step in the chain, and inputs this into a Gaussian likelihood along with the covariance and data vector, in order to sample the posterior. You can easily combine statistics (e.g. from different probes or redshift bins), omit certain scales, and apply priors.  