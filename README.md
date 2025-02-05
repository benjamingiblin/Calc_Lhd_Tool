# Calc_Lhd_Tool
## A one-stop code to perform MCMC or 2D-grid Bayesian parameter inference using a Gaussian process emulator in Python

This software package samples the posterior probability distributions either with a Markov Chain Monte Carlo (MCMC; for dim>2) or on a 2D grid.

The basic command to execute the software is:

```
# for an N-dimension MCMC sampling of the posterior:
python Calc_Lhd_MCMC.py <stats_param_file> <combos_param_file> <sys_param_file>		
# (see below for more info on the input param files)
```

Or

```
python Calc_Lhd_2D.py <parameter_file>			# for a 2D grid-based sampling of the posterior
```

## Example
```
stats_param_file=param_files_MCMC/cosmoSLICS_Mosaic/Sample_S8/stats/stats_params_4Chloe.dat
combs_param_file=param_files_MCMC/cosmoSLICS_Mosaic/Sample_S8/combs/combs_params_4Chloe.dat
python Calc_Lhd_MCMC.py $stats_param_file $combs_param_file
```
If...
```
Run_MCMC = True
```
...inside Calc_Lhd_MCMC.py, this will run a systematics-free MCMC sampling (Omega_m,S_8,h,w_0) using an emulator trained to predict the lensing PDF (stat explored in [Giblin et al. 2022][1]). If this is set to False, it will plot the contours made by a previous MCMC.


## Installation

Navigate to whether on your machine you would like to install the code and run:
```
git clone https://github.com/benjamingiblin/Calc_Lhd_Tool.git .
```

**Dependencies:** it is assumed that you have an up-to-date Anaconda distribution, which contains practically everything the code needs to run. If you encounter an error that a required python package is missing, this can easily be solved with:
```
pip install <package_name>
```

## More Info on MCMC Functionality

**The statistics paramfile...**

... lists the statistics which the emulator can read in and use to train the emulator. An example of a stat is the shear 2PCF measured in a given tomographic redshift bin. Each stat appears under its own banner (e.g. *---- STATISTIC 1 ----*).

Here are some of the key variables specified for each stat. Please note, there must **always be a space** between the variable name, the '=', and the argument which follows. 

   - nBins: number of bins used for this stat (e.g. theta bins for shear 2PCF).
   - Bins_To_Use: use this to make scale cuts (range(<nBins>)) will use all. 
   - PredFile: the address and general filename of training set predictions, e.g. PredictionXXXX.dat for predictions in separate 
files, formatted [x_array, Prediction] as columns. Alternatively, you can give all predictions as one pickled file with extension '.npy',such that Data[1,j,k] gives j'th prediction at k'th x-array element. Corresponding x-arrays are at element Data[0,j,k].
   - PredIDs: an array that replaces the 'XXXX' in TrainFile when reading in. Not used if TrainFile has '.npy' extension.
   - PredNodesFile: address of the nodes (often this is cosmol. params) corresponding to the training predicitons.
   - PredNodesCols: which columns in TrainNodesFile to use. Default is all.
   - DataFile: address containing the data vector for this stat to used in the likelihood.
   - DataCols: what columns of the DataFile to read (usually 0 is the x-array, 1 is the y-array). Note binning *must* match the PredFile predictions.
    - Transform: what transform to perform on the training predictions to make the emulation more accurate. Options are *log* or, e.g., *xy1e4* which means scale the predictions by x-array X 10,000 (theta X 10,000 in the case of 2PCF). *1e4* is arbitrary and any number could be specified here. All transformations are removed before emulator preds enter the likelihood. 
    - Perform_PCA: whether to also perform a principal component analysis on the training set preds, and if so...
    - n_components: how many PCA components to use.


**The combinations paramfile...**

It is possible to do arbitrary combinations of stats in the likelihood. E.g.
```
Combine_Stats = [range(76,91)]
```
will combine all stats listed in the stats paramfile from *STATISTIC 76* up to and including *STATISTIC 90*. These could for example be the shear correlation functions measured in different redshift bins.

This file contains the following variables:
     - DataNodesFile: file containing the true values of the parameters sampled in the MCMC (input cosmol. params for the data).
     - Plot_Limits: limits for the params used on the plot. 
     - Plot_Dims: what dimensions to plot (maybe you want to only plot some dimensions, neglecting nuisanace params).
     - Apply_Hartlap: If True will apply the [Hartlap 2007][3] correction to the covariance matrix.
     - nLabels: the axes labels to use on the contour plot.
     - savedirectory: where the results get saved.


Then further down, we have sections relating to combinations of statistics each under their own banner (e.g. *---- COMBINATION 1 ----* or e.g. *---- COMBINATION X ----*). Whichever section you specify as *COMBINATION 1* will be used for whatever combo of stats is specified first in the *Combine_Stats* array. You can specify multiple different combinations of stats in the *Combine_Stats* array, and then title the sections below, COMBINATION 1, COMBINATION 2,...etc., in order to run MCMCs for multiple combos one after the other, or to plot multiple chains you have already ran.
 






[1]: https://arxiv.org/abs/2211.05708
[2]: https://github.com/benjamingiblin/GPR_Emulator
[3]: https://arxiv.org/abs/astro-ph/0608064