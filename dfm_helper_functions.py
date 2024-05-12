from fredapi import Fred
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
from statsmodels.tsa.stattools import adfuller
from scipy.linalg import orth
from pprint import pprint
from pybea.client import BureauEconomicAnalysisClient
from datetime import date

def get_factor_pca(df_factor, n_factors = 1, plot_factors = False, return_names = False):
  '''
  takes in a df of multiple series and returns loadings and factors from PCA

  INPUT:
  df_factor = pd.DataFrame()
  n_factors = integer              # (default 1) if n_factors < 1 it will return the number of factors that explains that percentage of variance
  plot_factors = bool              # (default False) plot variance explained by n factors
  return_names = bool              # (default False) returns the name of the most important series in each factor

  OUTPUT:
  df_factor_values = pd.DataFrame()
  df_loadings      = pd.DataFrame()
  '''
  # drop any NaN values
  df_factor = df_factor.dropna(how='any',axis=0)

  # get the factor values and standardized the series
  X = df_factor.values
  standardize_series = StandardScaler() # demean and standardize to unit variance
  X_standardized     = standardize_series.fit_transform(X)

  # apply pca to X_standardized
  pca   = PCA(n_factors)
  X_pca = pca.fit(X_standardized)
  df_loadings = pd.DataFrame(pca.components_, columns = df_factor.columns)

  # get factor values from: f_i = (L'L)^{-1}L'(Y_{i} - y_{bar}), L in the loading matrix
  # X_standardized[-1] represents the factor values at time t
  df_factor_values = pd.DataFrame(X_standardized@np.transpose(pca.components_))
  df_factor_values.index = pd.DatetimeIndex(df_factor.index)
  #df_factor_values.columns = ["factor"]

  # plot the explained_var per component
  if plot_factors:
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');

  # getting the most important features from pca
  n_pcs= pca.n_components_
  most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
  initial_feature_names = df_factor.columns

  # get the most important feature names
  most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

  if return_names:
    return df_factor_values, df_loadings, most_important_names
  else:
    return df_factor_values, df_loadings
  
def repeat(arr, count):
  '''
  repeat an array n times

  INPUTS:
  arr   = np.array()
  count = int            # number of times you would like to repeat the array

  OUTPUT
  arr   = np.array()
  '''
  return np.stack([arr for _ in range(count)], axis=0)

def _Create_PPCA_Chart(Y, C, Ye, i_missing, iter):
  '''
  helper function called if plot == True in _Get_Factor_PPCA, Note this has been
  made for when the data used for PPCA is agg_sectors_inf and may contain some bugs.
  If the code breaks in a none obvious way just use plot = False in _Get_Factor_PPCA

  INPUTS:
  Y         = pd.DataFrame()
  C         = np.array()
  Ye        = np.array()
  i_missing = np.array()
  iter      = int

  OUTPUTS:
  plt.chart()
  '''

  # get norm of C to scale for every plot
  C_norm = np.linalg.norm(C)
  C_plot = C/C_norm

  # change labels and add legends
  plt.figure(figsize=(6,4))
  plt.xlabel(Y.columns[0])
  plt.ylabel(Y.columns[1])
  plt.title("Iteration: "+ str(iter))


  # plot missing values
  scatter_interp   = plt.scatter(Ye[i_missing[:,1],0], Ye[i_missing[:,1],1], color="grey")

  # plot realized values
  scatter_realized = plt.scatter(Ye[np.invert(i_missing[:,1]),0], Ye[np.invert(i_missing[:,1]),1], color="blue")

  # plot normalized C
  plot_C           = plt.plot(np.linspace(0, C_plot[0]/100, 10), np.linspace(0, C_plot[1]/100, 10), color='red')

  plt.legend([scatter_interp, scatter_realized, plot_C[0]], ['Interpolated y', 'Realized y', 'C (Normalized)'])

  plt.show()

def _Get_Factor_PPCA(Y, d=1, iter_bound=1E-3, plot=False, verbose=False,):
  '''
  creates factors with missing data using the EM algorithm
  Explaination of different elements of function:

  d:  ( int   ) number of latent dimensions
  ss: ( float ) isotropic variance outside subspace
  C:  (D by d ) C*C' + I*ss is covariance model, C has scaled principal directions as cols
  X:  (N by d ) expected states
  Ye: (N by D ) expected complete observations (differs from Y if data is missing)

  N, D      = shape(Y) #N observations in D dimensions (i.e. D is number of features, N is samples)
  i_missing = isnan(Y) indexes of missing observations
  N_missing = hidden.sum() #N of missing observations

  INPUTS:
  Y          = pd.DataFrame()
  d          = int                   # number of latent dimensions
  iter_bound = flt                   # float for convergence criteria
  plot       = bool                  # True if want to plot
  verbose    = bool                  # True if want to print each step

  OUTPUTS:
  X  = np.array()
  Ye = np.array()

  look at reference below for matlab code:
  J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
  '''
  C_array    = []

  N, D = Y.shape                      # N, number of observations. D, number of features                         # number of latent dimensions
  N_missing = Y.isna().sum().sum()    # find total number of missing values
  i_missing = np.array(Y.isna())      # save index of missing values

  # demean the data and replace NaN values with 0
  Ye = np.array(Y - Y.mean())
  Ye[i_missing] = 0

  # initialize variables for EM algo
  C   = np.random.normal(loc = 0, scale = 1, size = (D,d))  # standard normal for first iter
  CtC = C.T @ C
  X   = (Ye @ C) @ np.linalg.inv(CtC)
  X_proj_temp  = X @ C.T                                    # project X back into D to get initial ss
  X_proj_temp[i_missing] = 0                                # set 0 in same missing value indices
  ss = np.sum((X_proj_temp-Ye)**2) / (N*D - N_missing)

  iter = 1
  old  = np.Inf

  # EM algo
  while iter:
  ##########################################
    # E-step
    #########################################

    Sx     = np.linalg.inv(np.eye(d) + CtC/ss)
    ss_old = ss

    if N_missing > 0:
      X_proj = X @ C.T                    # project X back into D space
      Ye[i_missing] = X_proj[i_missing]   # best estimate for hidden values of Ye will be from X_proj

    X = (Ye @ C) @ (Sx/ss)

    ##########################################
    # M-step
    #########################################

    XtX = X.T @ X
    YtX = Ye.T @ X

    #update values
    C   = YtX @ np.linalg.inv(N_missing*Sx + XtX)
    CtC = C.T @ C
    ss  = (N_missing * np.sum(Sx*CtC) + np.sum((Ye - X @ C.T)**2) + N_missing * ss_old) / (N*D)
    Sx_det = np.linalg.det(Sx)

    objective = (1/2) * (-(N*D)*(1+np.log(ss)) - N * (np.trace(Sx)-np.log(Sx_det)) - np.trace(XtX) + N_missing*np.log(ss_old))

    obj_ch = np.abs( 1 - objective / old )
    old = objective

    if(verbose == True):
        print('Objective: %.2f, Relative Change %.5f' %(objective, obj_ch))
    C_array.append(C)

    if plot:
      # plotting the figures

      if iter == 1:
        _Create_PPCA_Chart(Y, C, Ye, i_missing, iter)

      if iter == 3:
        _Create_PPCA_Chart(Y, C, Ye, i_missing, iter)

      if iter == 50:
        _Create_PPCA_Chart(Y, C, Ye, i_missing, iter)

    iter = iter + 1
    if( obj_ch < iter_bound and iter > 100 ):
        iter = 0

  df_X = pd.DataFrame(X)
  df_X.set_index(Y.index)

  df_Ye = pd.DataFrame(Ye)
  df_Ye.set_index(Y.index)

  return df_X, df_Ye

def _Get_Agg_PCE_Window_Forecast(window_forecast_df, shares, monthly=False):
  '''
  get the Agg PCE forecast of the rolling window values by aggregating the 3 PCE sectors
  weighted by the shares

  INPUTS:
  window_forecast_df  = pd.DataFrame()      
  shares              = pd.DataFrame()

  OUTPUT:
  agg_window_forecast = pd.DataFrame()
  '''
  t_increment = -3         # go back 3 months if not monthly to get shares
  freq_str    = '3MS'

  if monthly:
    t_increment = -1
    freq_str    = 'MS'

  # we choose the shares to be the value at the time the forecast was made
  shares_start = str(window_forecast_df.index[0].date() + pd.DateOffset(months=t_increment))
  shares_end   = str(window_forecast_df.index[-1].date() + pd.DateOffset(months=t_increment))
  shares_dates = pd.date_range(start=shares_start, end=shares_end, freq=freq_str)

  shares = shares.loc[shares_dates]
  agg_window_forecast_df = window_forecast_df * shares.values
  
  return agg_window_forecast_df.sum(axis=1)
