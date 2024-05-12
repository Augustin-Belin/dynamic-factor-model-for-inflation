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
from datetime import datetime
import warnings
from scipy.optimize import minimize


# filter warnings for the time being 
warnings.filterwarnings("ignore")

class VAR:
  '''
  This class models a VAR
  '''
  def __init__(self, df_Y,  Y_lags, start_est_date='1999-01-01', end_est_date='2020-03-01'):
    self.df_Y           = df_Y
    self.Y_lags         = Y_lags
    self.T              = 25
    self.start_est_date = start_est_date
    self.end_est_date   = end_est_date

  def _Create_Lags(self, df, lags):
    '''
    this function takes in a pd.dataframe and returns a dataframe with the lagged data for a specified amount of lags
    as well as the original df that was passed

    INPUTS:
    df = pd.DataFrame()
    lags = integer  # specifing the number of lags

    OUTPUTS:
    df_org = pd.DataFrame() # original df passed
    df_lag = pd.DataFrame() # lags of df passed
    '''
    cols = df.shape[1]
    df_lag = np.zeros((len(df)-lags,cols*lags))

    # loop through all lags
    for lag in range(lags, 0, -1):
        df_lag[:,0+cols*(lags-lag):cols*(lags-lag+1)] = df.iloc[lag-1:(len(df)-lags+lag-1),:];

    # create copy of original df to cut first "lag" rows
    df_og = df.copy()
    df_og = df_og.iloc[lags:,:]

    # create index as dates and convert df_lag to df
    df_lag_dict = {'date': df_og.index}
    df_lag_dict = {}
    for series in df_og.columns:
        #retrieve series index
        col_index = df_og.columns.get_loc(series)

        # append and name all series with correct lags to df_lag
        for lag in range(lags):
            serie_lag = series+" lag "+str(lag+1)
            df_lag_dict[serie_lag] = df_lag[:, col_index*lags + lag]

    # create index as dates and convert df_lag to df
    df_lag = pd.DataFrame(df_lag_dict)
    df_lag.index = pd.DatetimeIndex(df_og.index)
    return df_og, df_lag

  def _Est_VAR(self, df_y, lags, df_exog=None, constant=1):
      '''
      INPUT:
      df_y = pd.DataFrame() # dataframe of dependent variables -- df_y must have a column name
      lags = integer        # corresponds to number of lags for df_y
      df_exog = pd.DataFrame() # dataframe of exogeneous variables (defaults to None)
      constant = bool       # specifies whether to include intercept in regression (defaults to 1)

      OUTPUT:
      results = dictionary  # within the dictionary there are many results linked to estimating a VAR
      results['x'] = np.array()                 # regressors
      results['y'] = np.array()                 # dependent var
      results['nvars'] = integer                # number of variables
      results['nobs'] = integer                 # number of observations
      results['ncoefs'] = integer               # number of coefficients
      results['beta'] = np.array()              # array of estimated betas NOTE: the final beta corresponds to the intercept
      results['sigma'] = np.array()             # covariance matrix
      results['vcovbeta'] = np.array()          # coefficient covariance matrix
      results['stdbeta'] = np.array()           # std of beta
      results['companion_mat'] = np.array()     # companion matrix
      results['aic'] = integer
      results['bic'] = integer
      results['hq'] = hq
      results['Yfit'] = np.array()              # fitted Y values
      results['SSR'] = integer                  # sum of squared residuals
      results['lags'] = integer                 # number of lags
      '''

      if lags == 0 and constant == 0:
          raise ValueError('WARNING: you are trying to estimate a VAR without a constant or lags')
      if lags > df_y.shape[0]:
          raise ValueError('WARNING: you must have more observations than lags')
      #if df_y.shape[0] < df_y.shape[1]:
       # print(df_y.shape[0])
        #print(df_y.shape[1])
          #raise ValueError('WARNING: You have more variables than observations or you have entered the data incorrectly.')

      # set estimation period
      start_est_date = self.start_est_date
      end_est_date   = self.end_est_date

      results = {}

      # initialize helper variables
      nvars = df_y.shape[1]
      nobs = df_y.shape[0] - lags
      ncoefs = nvars * lags + constant
      ibeg = lags + 1
      iend = df_y.shape[0]

      # initialize dependent variable and its lags
      z, xz = self._Create_Lags(df_y, lags)

      # restrict sample to precovid estimation period
      z  = z.loc[start_est_date:end_est_date]
      xz = xz.loc[start_est_date:end_est_date]

      if df_exog is not None:
          xz = pd.concat([xz, df_exog.loc[xz.index]], axis = 1)

      # add a constant term
      if constant:
          df_ones = pd.DataFrame(np.ones(xz.shape[0]))
          df_ones.set_index(xz.index, inplace=True)
          xz = pd.concat([xz, df_ones], axis = 1)
          #xz = pd.concat([xz, pd.DataFrame(np.ones(xz.shape[0]))], axis = 1)
          #xz = np.concatenate((xz, np.ones(xz.shape[0])), axis=1)

      beta = np.linalg.lstsq(xz, z, rcond=None)[0]
      Yfit = np.dot(xz, beta)
      uz = z - Yfit
      SSR = np.sum(uz**2)
      sigma = SSR / (len(uz) - lags * nvars + 1)
      vcovbeta = np.kron(sigma, np.linalg.inv(np.dot(xz.T, xz)))
      #SX = np.sqrt(np.diag(vcovbeta)).reshape(nvars, nvars)
      SX = 1
      companion_mat = np.vstack((beta[:lags * nvars, :].T, np.eye((lags - 1) * nvars, lags * nvars)))

      ldet = np.log(np.linalg.det(np.cov(uz)))
      nbc = beta.size
      aic = ldet + 2 * nbc / nobs
      bic = ldet + nbc * np.log(nobs) / nobs
      hq = ldet + 2 * nbc * np.log(np.log(nobs)) / nobs

      results['x'] = xz
      results['y'] = z
      results['nvars'] = nvars
      results['nobs'] = nobs
      results['ncoefs'] = ncoefs
      results['beta'] = beta
      results['sigma'] = sigma
      results['vcovbeta'] = vcovbeta
      results['stdbeta'] = SX
      results['companion_mat'] = companion_mat
      results['aic'] = aic
      results['bic'] = bic
      results['hq'] = hq
      results['Yfit'] = Yfit
      results['SSR'] = SSR
      results['lags'] = lags
      return results

  def _Get_Estimates(self):
    '''
    call multiple functions to get the parameters of the model
    '''
    model_params      = self._Est_VAR(self.df_Y, self.Y_lags)
    self.model_params = model_params

    return model_params

  def _Create_Forecast(self, forecast_date=None):
    '''
    Note: the model_params['beta'] arrary is a (N_vars * N_lags + N_factors + 1 ,N_vars) array
    for example with N_vars = 3, N_lags = 3, N_factors = 3

    |   Var1   |   Var1   |   Var1   |
    ----------------------------------
    |  Var1L1  |  Var1L1  |  Var1L1  |
    |  Var1L2  |  Var1L2  |  Var1L2  |
    |  Var1L3  |  Var1L3  |  Var1L3  |
    |  Var2L1  |  Var2L1  |  Var1L1  |
    |  Var2L2  |  Var2L2  |  Var2L2  |
    |  Var2L3  |  Var2L3  |  Var2L3  |
    |  Var3L1  |  Var3L1  |  Var3L1  |
    |  Var3L2  |  Var3L2  |  Var3L2  |
    |  Var3L3  |  Var3L3  |  Var3L3  |
    |  factor1 |  factor1 |  factor1 |
    |  factor2 |  factor2 |  factor2 |
    |  factor3 |  factor3 |  factor3 |
    |  cons.   |  cons.   |  cons.   |

    INPUTS:
    self.Y_df          = pd.DataFrame()    # contains historical data for the dependent variables
    self.df_exog       = pd.DataFrame()    # contains historical data for the factors
    self.Y_lags        = int               # number of lags of dependent variable
    self.T             = int               # forecast horizon
    self.betas         = np.array()        # of the form above
    self.forecast_date = str()             # if not specified defaults to current date

    OUTPUTS:
    Y_forecast_df = pd.DataFrame()  # (T, N_var) forecast for all dependent variables
    '''
    Y_df      = self.df_Y
    Y_lags    = self.Y_lags
    beta      = self.model_params["beta"]
    T         = self.T

    N_vars    = len(Y_df.columns)
    N_lags    = 3

    # restrict Y_df to have data up to forecast_date
    if forecast_date != None:
      Y_df    = Y_df.loc[:forecast_date]

    # get factor forecast up to time T and get forecast dates
    forecast_dates      = pd.date_range(start=Y_df.index[-1], periods=T+1, freq='MS')
    self.forecast_dates = forecast_dates

    # get each forecast t+1 by creating array below and multiplying by the betas array
    # [Var1L1, Var1L2, Var1L3, Var2L1, Var2L2, Var2L3, Var3L1, Var3L2, Var3L4, 1]

    for t in range(T):
      # initialize the array
      realized_array = np.zeros([N_vars * N_lags + 1,1])
      for i, y in enumerate(Y_df.columns):
        realized_array[i*N_lags:(i+1)*N_lags] = np.array(Y_df[y].iloc[-N_lags:]).reshape(-1,1)

      # add the constant to the realized array
      realized_array[-1] = 1

      # get forecast for the current period
      forecast_array   = np.dot(realized_array.T, beta)
      ts = pd.to_datetime(forecast_dates[t+1], format="%Y-%m-%d %H:%M:%S.%f")     # get timestamp of t+i
      new_row = pd.DataFrame(forecast_array, columns=Y_df.columns, index=[ts])
      Y_df = pd.concat([Y_df, pd.DataFrame(new_row)], ignore_index=False)         # concat with previous row and iterate

    self.Y_forecast_df = Y_df.iloc[-T:]

    return Y_df.iloc[-T:]

  def _Check_Model_Performance(self, window_start="01-01-2010", window_end="01-01-2020", monthly=False):
    '''
    check the 1 quarter ahead out of sample forecast of model using rolling window estimation

    INPUTS:
    window_start = str()         # date when we start the rolling window estimation
    window_end   = str()         # date when we end the rolling window estimation
    monthly      = bool          # True if want monthly forecasts otherwise defaults to quarterly

    OUTPUTS:
    self.window_forecast  = pd.DataFrame()       # 3 or 1 month ahead forecasts
    self.window_errors    = pd.DataFrame()       # 3 or 1 month ahead forecast errors
    '''
    window_forecast = []
    window_dates = pd.date_range(start=window_start, end=window_end, freq='MS')

    t_increment = 1   # how much we increase rolling window by

    if not monthly:
      window_dates = pd.date_range(start=window_start, end=window_end, freq='3MS')
      t_increment  = 3

    for date in window_dates:
      self.start_est_date = str((datetime.strptime(self.start_est_date, '%Y-%m-%d').date() + pd.DateOffset(months=t_increment)).date())
      self.end_est_date   = date                   # change estimation date over the window
      self._Get_Estimates()                        # update parameters
      self._Create_Forecast(forecast_date=date)    # get forecast over window

      if not monthly:
        window_forecast.append(self.Y_forecast_df.iloc[2])   # get values 3 months ahead
      else:
        window_forecast.append(self.Y_forecast_df.iloc[0])   # get values 1 month ahead

    window_forecast = pd.DataFrame(window_forecast)
    window_errors   = window_forecast - self.df_Y.loc[window_forecast.index]

    self.window_forecast = window_forecast
    self.window_errors   = window_errors



class DFM_PCA:

  '''
  This class models a DFM using factors estimated by PCA and other endogenous regressors
  '''
  def __init__(self, df_Y, factor_dict, Y_lags, start_est_date='1999-01-01', end_est_date='2020-03-01', quarterly =0 , monthly =1, daily =1):
    self.df_Y           = df_Y
    self.factor_dict    = factor_dict
    self.Y_lags         = Y_lags
    self.T              = 25
    self.start_est_date = start_est_date
    self.end_est_date   = end_est_date

  def _Create_Lags(self, df, lags):
    '''
    this function takes in a pd.dataframe and returns a dataframe with the lagged data for a specified amount of lags
    as well as the original df that was passed

    INPUTS:
    df = pd.DataFrame()
    lags = integer  # specifing the number of lags

    OUTPUTS:
    df_org = pd.DataFrame() # original df passed
    df_lag = pd.DataFrame() # lags of df passed
    '''
    cols = df.shape[1]
    df_lag = np.zeros((len(df)-lags,cols*lags))

    # loop through all lags
    for lag in range(lags, 0, -1):
        df_lag[:,0+cols*(lags-lag):cols*(lags-lag+1)] = df.iloc[lag-1:(len(df)-lags+lag-1),:];

    # create copy of original df to cut first "lag" rows
    df_og = df.copy()
    df_og = df_og.iloc[lags:,:]

    # create index as dates and convert df_lag to df
    df_lag_dict = {'date': df_og.index}
    df_lag_dict = {}
    for series in df_og.columns:
        #retrieve series index
        col_index = df_og.columns.get_loc(series)

        # append and name all series with correct lags to df_lag
        for lag in range(lags):
            serie_lag = series+" lag "+str(lag+1)
            df_lag_dict[serie_lag] = df_lag[:, col_index*lags + lag]

    # create index as dates and convert df_lag to df
    df_lag = pd.DataFrame(df_lag_dict)
    df_lag.index = pd.DatetimeIndex(df_og.index)
    return df_og, df_lag

  def _Get_Factor_PCA(self, df_factor, n_factors = 1, plot_factors = False, return_names = False):
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

  def _Get_Factors(self, factor_dict):
    '''
    unpack data dict into a dict of factors and loadings

    INPUTS:
    factor_dict          = dict

    OUTPUT:
    dict_factor_matrix   = dict
    dict_loadings_matrix = dict
    '''
    # initialize dictionaries for factors and loadings
    dict_factor_matrix   = {}
    dict_loadings_matrix = {}

    # unpack factor dictionary to be able to pass through the VAR
    for factor_name in self.factor_dict:
      factor_tmp, loadings_tmp = self._Get_Factor_PCA(self.factor_dict[factor_name]);
      dict_factor_matrix[factor_name]   = factor_tmp
      dict_loadings_matrix[factor_name] = loadings_tmp

    self.dict_factor_matrix   = dict_factor_matrix
    self.dict_loadings_matrix = dict_loadings_matrix

    return dict_factor_matrix, dict_loadings_matrix

  def _Est_VAR(self, df_y, lags, df_exog=None, constant=1):
      '''
      INPUT:
      df_y = pd.DataFrame() # dataframe of dependent variables -- df_y must have a column name
      lags = integer        # corresponds to number of lags for df_y
      df_exog = pd.DataFrame() # dataframe of exogeneous variables (defaults to None)
      constant = bool       # specifies whether to include intercept in regression (defaults to 1)

      OUTPUT:
      results = dictionary  # within the dictionary there are many results linked to estimating a VAR
      results['x'] = np.array()                 # regressors
      results['y'] = np.array()                 # dependent var
      results['nvars'] = integer                # number of variables
      results['nobs'] = integer                 # number of observations
      results['ncoefs'] = integer               # number of coefficients
      results['beta'] = np.array()              # array of estimated betas NOTE: the final beta corresponds to the intercept
      results['sigma'] = np.array()             # covariance matrix
      results['vcovbeta'] = np.array()          # coefficient covariance matrix
      results['stdbeta'] = np.array()           # std of beta
      results['companion_mat'] = np.array()     # companion matrix
      results['aic'] = integer
      results['bic'] = integer
      results['hq'] = hq
      results['Yfit'] = np.array()              # fitted Y values
      results['SSR'] = integer                  # sum of squared residuals
      results['lags'] = integer                 # number of lags
      '''

      if lags == 0 and constant == 0:
          raise ValueError('WARNING: you are trying to estimate a VAR without a constant or lags')
      if lags > df_y.shape[0]:
          raise ValueError('WARNING: you must have more observations than lags')
      #if df_y.shape[0] < df_y.shape[1]:
       # print(df_y.shape[0])
        #print(df_y.shape[1])
          #raise ValueError('WARNING: You have more variables than observations or you have entered the data incorrectly.')

      # set estimation period
      start_est_date = self.start_est_date
      end_est_date   = self.end_est_date

      results = {}

      # initialize helper variables
      nvars = df_y.shape[1]
      nobs = df_y.shape[0] - lags
      ncoefs = nvars * lags + len(df_exog.columns) + constant
      ibeg = lags + 1
      iend = df_y.shape[0]

      # initialize dependent variable and its lags
      z, xz = self._Create_Lags(df_y, lags)

      # restrict sample to precovid estimation period
      z  = z.loc[start_est_date:end_est_date]
      xz = xz.loc[start_est_date:end_est_date]

      if df_exog is not None:
          xz = pd.concat([xz, df_exog.loc[xz.index]], axis = 1)

      # add a constant term
      if constant:
          df_ones = pd.DataFrame(np.ones(xz.shape[0]))
          df_ones.set_index(xz.index, inplace=True)
          xz = pd.concat([xz, df_ones], axis = 1)
          #xz = pd.concat([xz, pd.DataFrame(np.ones(xz.shape[0]))], axis = 1)
          #xz = np.concatenate((xz, np.ones(xz.shape[0])), axis=1)

      beta = np.linalg.lstsq(xz, z, rcond=None)[0]
      Yfit = np.dot(xz, beta)
      uz = z - Yfit
      SSR = np.sum(uz**2)
      sigma = SSR / (len(uz) - lags * nvars + 1)
      vcovbeta = np.kron(sigma, np.linalg.inv(np.dot(xz.T, xz)))
      #SX = np.sqrt(np.diag(vcovbeta)).reshape(nvars, nvars)
      SX = 1
      companion_mat = np.vstack((beta[:lags * nvars, :].T, np.eye((lags - 1) * nvars, lags * nvars)))

      ldet = np.log(np.linalg.det(np.cov(uz)))
      nbc = beta.size
      aic = ldet + 2 * nbc / nobs
      bic = ldet + nbc * np.log(nobs) / nobs
      hq = ldet + 2 * nbc * np.log(np.log(nobs)) / nobs

      results['x'] = xz
      results['y'] = z
      results['nvars'] = nvars
      results['nobs'] = nobs
      results['ncoefs'] = ncoefs
      results['beta'] = beta
      results['sigma'] = sigma
      results['vcovbeta'] = vcovbeta
      results['stdbeta'] = SX
      results['companion_mat'] = companion_mat
      results['aic'] = aic
      results['bic'] = bic
      results['hq'] = hq
      results['Yfit'] = Yfit
      results['SSR'] = SSR
      results['lags'] = lags
      return results

  def _Get_Estimates(self):
    '''
    call multiple functions to get the parameters of the model
    '''
    exog, pca_loadings = self._Get_Factors(self.factor_dict)
    df_exog = pd.DataFrame()

    # create the exog df
    for factor in exog:
      df_exog[factor] = exog[factor]

    model_params      = self._Est_VAR(self.df_Y, self.Y_lags, df_exog)
    self.model_params = model_params
    self.df_exog      = df_exog

    return model_params

  def _Get_Factor_Forecast(self, factor_df, T, forecast_date=None):
    '''
    get the factor forcast up to time T

    INPUTS:
    factor_df      = pd.DataFrame()   # this would be the same as exog_df
    T              = int              # horizon of what you would like to forcast to
    forecast_date  = str()            # start date of forecast defaults to current date

    OUTPUTS:
    forecast_factor_df = pd.DataFrame()  # size (T, N_factors)
    '''
    start_est_date = self.start_est_date
    end_est_date   = self.end_est_date

    # restrict extimation of factor persistence to subsample
    factor_df_regression = factor_df.loc[start_est_date:end_est_date]
    rho_dict  = {}

    # performing regression F_t = rho F_t_1 for each factor
    for factor in factor_df_regression:
      F_t   = np.array(factor_df_regression[factor].iloc[1:]).reshape(-1,1)      # skip first observation
      F_t_1 = np.array(factor_df_regression[factor].iloc[:-1]).reshape(-1,1)     # skip last observation
      rho   = np.linalg.lstsq(F_t_1, F_t, rcond=None)[0]
      rho_dict[factor] = rho.item()

    # check to see if any of the data has not come and iterate forward to get our best estimate
    for factor in factor_df:
      last_realized_month = factor_df[factor].last_valid_index()                               # get timestamp of most recent realized
      t_fill = len(factor_df[factor]) - len(factor_df[factor].loc[:last_realized_month])       # get t we need to iterate forward on
      if t_fill > 0:
        last_realized_i = len(factor_df[factor].loc[:last_realized_month])
        for t in range(t_fill):
          factor_df[factor].iloc[last_realized_i] = rho_dict[factor] * factor_df[factor].iloc[last_realized_i - 1]


    # now create df that has forecasts of factors
    factor_forecast_dict = {}

    # restrict factor_df depending on what dates we want the forecast for
    if forecast_date != None:
      factor_df = factor_df.loc[:forecast_date,:]

    for factor in factor_df:
      factor_forecast_array = []
      f_t = factor_df[factor].iloc[-1]                    # initalize f_t with most realized value
      for t in range(T):
        f_t_1 = rho_dict[factor] * f_t                    # iterate forward and append
        factor_forecast_array.append(f_t_1)
        f_t = f_t_1
      factor_forecast_dict[factor] = factor_forecast_array
    # get forecast dates
    most_realized_date = factor_df.last_valid_index()
    forecast_dates     = pd.date_range(start=most_realized_date, periods=T+1, freq='MS')

    # create dataframe
    factor_forecast_df = pd.DataFrame(factor_forecast_dict)
    factor_forecast_df.set_index(forecast_dates[1:], inplace=True)

    # concate dataframe so that we can later choose the correct data from forecast dates
    factor_forecast_df = pd.concat([factor_df, factor_forecast_df])

    self.factor_forecast_df = factor_forecast_df

    return factor_forecast_df

  def _Create_Forecast(self, forecast_date=None):
    '''
    Note: the model_params['beta'] arrary is a (N_vars * N_lags + N_factors + 1 ,N_vars) array
    for example with N_vars = 3, N_lags = 3, N_factors = 3

    |   Var1   |   Var1   |   Var1   |
    ----------------------------------
    |  Var1L1  |  Var1L1  |  Var1L1  |
    |  Var1L2  |  Var1L2  |  Var1L2  |
    |  Var1L3  |  Var1L3  |  Var1L3  |
    |  Var2L1  |  Var2L1  |  Var1L1  |
    |  Var2L2  |  Var2L2  |  Var2L2  |
    |  Var2L3  |  Var2L3  |  Var2L3  |
    |  Var3L1  |  Var3L1  |  Var3L1  |
    |  Var3L2  |  Var3L2  |  Var3L2  |
    |  Var3L3  |  Var3L3  |  Var3L3  |
    |  factor1 |  factor1 |  factor1 |
    |  factor2 |  factor2 |  factor2 |
    |  factor3 |  factor3 |  factor3 |
    |  cons.   |  cons.   |  cons.   |

    INPUTS:
    self.Y_df          = pd.DataFrame()    # contains historical data for the dependent variables
    self.df_exog       = pd.DataFrame()    # contains historical data for the factors
    self.Y_lags        = int               # number of lags of dependent variable
    self.T             = int               # forecast horizon
    self.betas         = np.array()        # of the form above
    self.forecast_date = str()             # if not specified defaults to current date

    OUTPUTS:
    Y_forecast_df = pd.DataFrame()  # (T, N_var) forecast for all dependent variables
    '''
    Y_df      = self.df_Y
    factor_df = self.df_exog
    Y_lags    = self.Y_lags
    beta      = self.model_params["beta"]
    T         = self.T

    N_vars    = len(Y_df.columns)
    N_lags    = 3
    N_factors = len(factor_df.columns)

    # restrict Y_df to have data up to forecast_date
    if forecast_date != None:
      Y_df    = Y_df.loc[:forecast_date]

    # get factor forecast up to time T and get forecast dates
    forecast_dates      = pd.date_range(start=Y_df.index[-1], periods=T+1, freq='MS')
    self.forecast_dates = forecast_dates
    factor_forecast_df  = self._Get_Factor_Forecast(factor_df, T, forecast_date)      # get factor forecast
    factor_forecast_df  = factor_forecast_df.loc[forecast_dates]


    # get each forecast t+1 by creating array below and multiplying by the betas array
    # [Var1L1, Var1L2, Var1L3, Var2L1, Var2L2, Var2L3, Var3L1, Var3L2, Var3L4, factor1, factor2, factor3, 1]

    for t in range(T):
      # initialize the array
      realized_array = np.zeros([N_vars * N_lags + N_factors + 1,1])
      for i, y in enumerate(Y_df.columns):
        realized_array[i*N_lags:(i+1)*N_lags] = np.array(Y_df[y].iloc[-N_lags:]).reshape(-1,1)

      # add the factor to the realized array
      realized_array[(N_vars)*N_lags:(N_vars)*N_lags + N_factors] = np.array(factor_forecast_df.iloc[t,:]).T.reshape(-1,1)

      # add the constant to the realized array
      realized_array[-1] = 1

      # get forecast for the current period
      forecast_array   = np.dot(realized_array.T, beta)
      ts = pd.to_datetime(forecast_dates[t+1], format="%Y-%m-%d %H:%M:%S.%f")     # get timestamp of t+i
      new_row = pd.DataFrame(forecast_array, columns=Y_df.columns, index=[ts])
      Y_df = pd.concat([Y_df, pd.DataFrame(new_row)], ignore_index=False)         # concat with previous row and iterate

    self.Y_forecast_df = Y_df.iloc[-T:]
    self.factor_forecast_df = factor_forecast_df

    return Y_df.iloc[-T:]

  def _Check_Model_Performance(self, window_start="01-01-2010", window_end="01-01-2020", monthly=False):
    '''
    check the 1 quarter ahead out of sample forecast of model using rolling window estimation

    INPUTS:
    window_start = str()         # date when we start the rolling window estimation
    window_end   = str()         # date when we end the rolling window estimation
    monthly      = bool          # True if want monthly forecasts otherwise defaults to quarterly

    OUTPUTS:
    self.window_forecast  = pd.DataFrame()       # 3 or 1 month ahead forecasts
    self.window_errors    = pd.DataFrame()       # 3 or 1 month ahead forecast errors
    '''
    window_forecast = []
    window_dates = pd.date_range(start=window_start, end=window_end, freq='MS')

    t_increment = 1   # how much we increase rolling window by

    if not monthly:
      window_dates = pd.date_range(start=window_start, end=window_end, freq='3MS')
      t_increment  = 3

    for date in window_dates:
      self.start_est_date = str((datetime.strptime(self.start_est_date, '%Y-%m-%d').date() + pd.DateOffset(months=t_increment)).date())
      self.end_est_date   = date                   # change estimation date over the window
      self._Get_Estimates()                        # update parameters
      self._Create_Forecast(forecast_date=date)    # get forecast over window

      if not monthly:
        window_forecast.append(self.Y_forecast_df.iloc[2])   # get values 3 months ahead
      else:
        window_forecast.append(self.Y_forecast_df.iloc[0])   # get values 1 month ahead

    window_forecast = pd.DataFrame(window_forecast)
    window_errors   = window_forecast - self.df_Y.loc[window_forecast.index]

    self.window_forecast = window_forecast
    self.window_errors = window_errors

class DFM_PPCA:

  '''
  This class models a DFM using factors estimated by PPCA and other endogenous regressors
  This class deals with missing data through the EM algorithem.
  '''
  def __init__(self, df_Y, factor_dict, Y_lags, start_est_date='1999-01-01', end_est_date='2020-03-01', quarterly =0 , monthly =1, daily =1):
    self.df_Y           = df_Y
    self.factor_dict    = factor_dict
    self.Y_lags         = Y_lags
    self.T              = 25
    self.start_est_date = start_est_date
    self.end_est_date   = end_est_date

  def _Create_Lags(self, df, lags):
    '''
    this function takes in a pd.dataframe and returns a dataframe with the lagged data for a specified amount of lags
    as well as the original df that was passed

    INPUTS:
    df = pd.DataFrame()
    lags = integer  # specifing the number of lags

    OUTPUTS:
    df_org = pd.DataFrame() # original df passed
    df_lag = pd.DataFrame() # lags of df passed
    '''
    cols = df.shape[1]
    df_lag = np.zeros((len(df)-lags,cols*lags))

    # loop through all lags
    for lag in range(lags, 0, -1):
        df_lag[:,0+cols*(lags-lag):cols*(lags-lag+1)] = df.iloc[lag-1:(len(df)-lags+lag-1),:];

    # create copy of original df to cut first "lag" rows
    df_og = df.copy()
    df_og = df_og.iloc[lags:,:]

    # create index as dates and convert df_lag to df
    df_lag_dict = {'date': df_og.index}
    df_lag_dict = {}
    for series in df_og.columns:
        #retrieve series index
        col_index = df_og.columns.get_loc(series)

        # append and name all series with correct lags to df_lag
        for lag in range(lags):
            serie_lag = series+" lag "+str(lag+1)
            df_lag_dict[serie_lag] = df_lag[:, col_index*lags + lag]

    # create index as dates and convert df_lag to df
    df_lag = pd.DataFrame(df_lag_dict)
    df_lag.index = pd.DatetimeIndex(df_og.index)
    return df_og, df_lag

  def _Get_Factor_PPCA(self, Y, d=1, iter_bound=1E-3, plot=False, verbose=False,):
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
    Y_std = np.array(np.std(Y)).T
    Ye    = np.array(Y - Y.mean())/Y_std
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

    X = X.reshape(-1)
    df_X = pd.Series(X, index = Y.index)

    df_Ye = pd.DataFrame(Ye)
    df_Ye.set_index(Y.index)

    return df_X, df_Ye

  def _Get_Factors(self, factor_dict):
    '''
    unpack data dict into a dict of factors and loadings

    INPUTS:
    factor_dict          = dict

    OUTPUT:
    dict_factor_matrix   = dict
    dict_interp_y        = dict    # has the interpolated y values from PPCA
    '''
    # initialize dictionaries for factors and loadings
    dict_factor_matrix   = {}
    dict_interp_y = {}

    # unpack factor dictionary to be able to pass through the VAR
    for factor_name in self.factor_dict:
      factor_tmp, interp_y_tmp = self._Get_Factor_PPCA(self.factor_dict[factor_name]);
      dict_factor_matrix[factor_name]   = factor_tmp
      dict_interp_y[factor_name] = interp_y_tmp

    self.dict_factor_matrix   = dict_factor_matrix
    self.dict_interp_y = dict_interp_y

    return dict_factor_matrix, dict_interp_y

  def _Est_VAR(self, df_y, lags, df_exog=None, constant=1):
      '''
      INPUT:
      df_y = pd.DataFrame() # dataframe of dependent variables -- df_y must have a column name
      lags = integer        # corresponds to number of lags for df_y
      df_exog = pd.DataFrame() # dataframe of exogeneous variables (defaults to None)
      constant = bool       # specifies whether to include intercept in regression (defaults to 1)

      OUTPUT:
      results = dictionary  # within the dictionary there are many results linked to estimating a VAR
      results['x'] = np.array()                 # regressors
      results['y'] = np.array()                 # dependent var
      results['nvars'] = integer                # number of variables
      results['nobs'] = integer                 # number of observations
      results['ncoefs'] = integer               # number of coefficients
      results['beta'] = np.array()              # array of estimated betas NOTE: the final beta corresponds to the intercept
      results['sigma'] = np.array()             # covariance matrix
      results['vcovbeta'] = np.array()          # coefficient covariance matrix
      results['stdbeta'] = np.array()           # std of beta
      results['companion_mat'] = np.array()     # companion matrix
      results['aic'] = integer
      results['bic'] = integer
      results['hq'] = hq
      results['Yfit'] = np.array()              # fitted Y values
      results['SSR'] = integer                  # sum of squared residuals
      results['lags'] = integer                 # number of lags
      '''

      if lags == 0 and constant == 0:
          raise ValueError('WARNING: you are trying to estimate a VAR without a constant or lags')
      if lags > df_y.shape[0]:
          raise ValueError('WARNING: you must have more observations than lags')
      #if df_y.shape[0] < df_y.shape[1]:
       # print(df_y.shape[0])
        #print(df_y.shape[1])
          #raise ValueError('WARNING: You have more variables than observations or you have entered the data incorrectly.')

      # set estimation period
      start_est_date = self.start_est_date
      end_est_date   = self.end_est_date

      results = {}

      # initialize helper variables
      nvars = df_y.shape[1]
      nobs = df_y.shape[0] - lags
      ncoefs = nvars * lags + len(df_exog.columns) + constant
      ibeg = lags + 1
      iend = df_y.shape[0]

      # initialize dependent variable and its lags
      z, xz = self._Create_Lags(df_y, lags)

      # restrict sample to precovid estimation period
      z  = z.loc[start_est_date:end_est_date]
      xz = xz.loc[start_est_date:end_est_date]

      if df_exog is not None:
          xz = pd.concat([xz, df_exog.loc[xz.index]], axis = 1)

      # add a constant term
      if constant:
          df_ones = pd.DataFrame(np.ones(xz.shape[0]))
          df_ones.set_index(xz.index, inplace=True)
          xz = pd.concat([xz, df_ones], axis = 1)
          #xz = pd.concat([xz, pd.DataFrame(np.ones(xz.shape[0]))], axis = 1)
          #xz = np.concatenate((xz, np.ones(xz.shape[0])), axis=1)

      beta = np.linalg.lstsq(xz, z, rcond=None)[0]
      Yfit = np.dot(xz, beta)
      uz = z - Yfit
      SSR = np.sum(uz**2)
      sigma = SSR / (len(uz) - lags * nvars + 1)
      vcovbeta = np.kron(sigma, np.linalg.inv(np.dot(xz.T, xz)))
      #SX = np.sqrt(np.diag(vcovbeta)).reshape(nvars, nvars)
      SX = 1
      companion_mat = np.vstack((beta[:lags * nvars, :].T, np.eye((lags - 1) * nvars, lags * nvars)))

      ldet = np.log(np.linalg.det(np.cov(uz)))
      nbc = beta.size
      aic = ldet + 2 * nbc / nobs
      bic = ldet + nbc * np.log(nobs) / nobs
      hq = ldet + 2 * nbc * np.log(np.log(nobs)) / nobs

      results['x'] = xz
      results['y'] = z
      results['nvars'] = nvars
      results['nobs'] = nobs
      results['ncoefs'] = ncoefs
      results['beta'] = beta
      results['sigma'] = sigma
      results['vcovbeta'] = vcovbeta
      results['stdbeta'] = SX
      results['companion_mat'] = companion_mat
      results['aic'] = aic
      results['bic'] = bic
      results['hq'] = hq
      results['Yfit'] = Yfit
      results['SSR'] = SSR
      results['lags'] = lags
      return results

  def _Get_Estimates(self):
    '''
    call multiple functions to get the parameters of the model
    '''
    exog, interp_y = self._Get_Factors(self.factor_dict)
    df_exog = pd.DataFrame()

    # create the exog df
    for factor in exog:
      df_exog[factor] = exog[factor]
    self.df_exog      = df_exog

    model_params      = self._Est_VAR(self.df_Y, self.Y_lags, df_exog)
    self.model_params = model_params
    self.df_exog      = df_exog

    return model_params

  def _Get_Factor_Forecast(self, factor_df, T, forecast_date=None):
    '''
    get the factor forcast up to time T

    INPUTS:
    factor_df      = pd.DataFrame()   # this would be the same as exog_df
    T              = int              # horizon of what you would like to forcast to
    forecast_date  = str()            # start date of forecast defaults to current date

    OUTPUTS:
    forecast_factor_df = pd.DataFrame()  # size (T, N_factors)
    '''
    start_est_date = self.start_est_date
    end_est_date   = self.end_est_date

    # restrict extimation of factor persistence to subsample
    factor_df_regression = factor_df.loc[start_est_date:end_est_date]
    rho_dict  = {}

    # performing regression F_t = rho F_t_1 for each factor
    for factor in factor_df_regression:
      F_t   = np.array(factor_df_regression[factor].iloc[1:]).reshape(-1,1)      # skip first observation
      F_t_1 = np.array(factor_df_regression[factor].iloc[:-1]).reshape(-1,1)     # skip last observation
      rho   = np.linalg.lstsq(F_t_1, F_t, rcond=None)[0]
      rho_dict[factor] = rho.item()

    # check to see if any of the data has not come and iterate forward to get our best estimate
    for factor in factor_df:
      last_realized_month = factor_df[factor].last_valid_index()                               # get timestamp of most recent realized
      t_fill = len(factor_df[factor]) - len(factor_df[factor].loc[:last_realized_month])       # get t we need to iterate forward on
      if t_fill > 0:
        last_realized_i = len(factor_df[factor].loc[:last_realized_month])
        for t in range(t_fill):
          factor_df[factor].iloc[last_realized_i] = rho_dict[factor] * factor_df[factor].iloc[last_realized_i - 1]


    # now create df that has forecasts of factors
    factor_forecast_dict = {}

    # restrict factor_df depending on what dates we want the forecast for
    if forecast_date != None:
      factor_df = factor_df.loc[:forecast_date,:]

    for factor in factor_df:
      factor_forecast_array = []
      f_t = factor_df[factor].iloc[-1]                    # initalize f_t with most realized value
      for t in range(T):
        f_t_1 = rho_dict[factor] * f_t                    # iterate forward and append
        factor_forecast_array.append(f_t_1)
        f_t = f_t_1
      factor_forecast_dict[factor] = factor_forecast_array
    # get forecast dates
    most_realized_date = factor_df.last_valid_index()
    forecast_dates     = pd.date_range(start=most_realized_date, periods=T+1, freq='MS')

    # create dataframe
    factor_forecast_df = pd.DataFrame(factor_forecast_dict)
    factor_forecast_df.set_index(forecast_dates[1:], inplace=True)

    # concate dataframe so that we can later choose the correct data from forecast dates
    factor_forecast_df = pd.concat([factor_df, factor_forecast_df])

    self.factor_forecast_df = factor_forecast_df

    return factor_forecast_df

  def _Create_Forecast(self, forecast_date=None):
    '''
    Note: the model_params['beta'] arrary is a (N_vars * N_lags + N_factors + 1 ,N_vars) array
    for example with N_vars = 3, N_lags = 3, N_factors = 3

    |   Var1   |   Var1   |   Var1   |
    ----------------------------------
    |  Var1L1  |  Var1L1  |  Var1L1  |
    |  Var1L2  |  Var1L2  |  Var1L2  |
    |  Var1L3  |  Var1L3  |  Var1L3  |
    |  Var2L1  |  Var2L1  |  Var1L1  |
    |  Var2L2  |  Var2L2  |  Var2L2  |
    |  Var2L3  |  Var2L3  |  Var2L3  |
    |  Var3L1  |  Var3L1  |  Var3L1  |
    |  Var3L2  |  Var3L2  |  Var3L2  |
    |  Var3L3  |  Var3L3  |  Var3L3  |
    |  factor1 |  factor1 |  factor1 |
    |  factor2 |  factor2 |  factor2 |
    |  factor3 |  factor3 |  factor3 |
    |  cons.   |  cons.   |  cons.   |

    INPUTS:
    self.Y_df          = pd.DataFrame()    # contains historical data for the dependent variables
    self.df_exog       = pd.DataFrame()    # contains historical data for the factors
    self.Y_lags        = int               # number of lags of dependent variable
    self.T             = int               # forecast horizon
    self.betas         = np.array()        # of the form above
    self.forecast_date = str()             # if not specified defaults to current date

    OUTPUTS:
    Y_forecast_df = pd.DataFrame()  # (T, N_var) forecast for all dependent variables
    '''
    Y_df      = self.df_Y
    factor_df = self.df_exog
    Y_lags    = self.Y_lags
    beta      = self.model_params["beta"]
    T         = self.T

    N_vars    = len(Y_df.columns)
    N_lags    = 3
    N_factors = len(factor_df.columns)

    # restrict Y_df to have data up to forecast_date
    if forecast_date != None:
      Y_df    = Y_df.loc[:forecast_date]

    # get factor forecast up to time T and get forecast dates
    forecast_dates      = pd.date_range(start=Y_df.index[-1], periods=T+1, freq='MS')
    self.forecast_dates = forecast_dates
    factor_forecast_df  = self._Get_Factor_Forecast(factor_df, T, forecast_date)      # get factor forecast
    factor_forecast_df  = factor_forecast_df.loc[forecast_dates]


    # get each forecast t+1 by creating array below and multiplying by the betas array
    # [Var1L1, Var1L2, Var1L3, Var2L1, Var2L2, Var2L3, Var3L1, Var3L2, Var3L4, factor1, factor2, factor3, 1]

    for t in range(T):
      # initialize the array
      realized_array = np.zeros([N_vars * N_lags + N_factors + 1,1])
      for i, y in enumerate(Y_df.columns):
        realized_array[i*N_lags:(i+1)*N_lags] = np.array(Y_df[y].iloc[-N_lags:]).reshape(-1,1)

      # add the factor to the realized array
      realized_array[(N_vars)*N_lags:(N_vars)*N_lags + N_factors] = np.array(factor_forecast_df.iloc[t,:]).T.reshape(-1,1)

      # add the constant to the realized array
      realized_array[-1] = 1

      # get forecast for the current period
      forecast_array   = np.dot(realized_array.T, beta)
      ts = pd.to_datetime(forecast_dates[t+1], format="%Y-%m-%d %H:%M:%S.%f")     # get timestamp of t+i
      new_row = pd.DataFrame(forecast_array, columns=Y_df.columns, index=[ts])
      Y_df = pd.concat([Y_df, pd.DataFrame(new_row)], ignore_index=False)         # concat with previous row and iterate

    self.Y_forecast_df = Y_df.iloc[-T:]
    self.factor_forecast_df = factor_forecast_df

    return Y_df.iloc[-T:]

  def _Check_Model_Performance(self, window_start="01-01-2010", window_end="01-01-2020", monthly=False):
    '''
    check the 1 quarter ahead out of sample forecast of model using rolling window estimation

    INPUTS:
    window_start = str()         # date when we start the rolling window estimation
    window_end   = str()         # date when we end the rolling window estimation
    monthly      = bool          # True if want monthly forecasts otherwise defaults to quarterly

    OUTPUTS:
    self.window_forecast  = pd.DataFrame()       # 3 or 1 month ahead forecasts
    self.window_errors    = pd.DataFrame()       # 3 or 1 month ahead forecast errors
    '''
    window_forecast = []
    window_dates = pd.date_range(start=window_start, end=window_end, freq='MS')

    t_increment = 1   # how much we increase rolling window by

    if not monthly:
      window_dates = pd.date_range(start=window_start, end=window_end, freq='3MS')
      t_increment  = 3

    for date in window_dates:
      self.start_est_date = str((datetime.strptime(self.start_est_date, '%Y-%m-%d').date() + pd.DateOffset(months=t_increment)).date())
      self.end_est_date   = date                   # change estimation date over the window
      self._Get_Estimates()                        # update parameters
      self._Create_Forecast(forecast_date=date)    # get forecast over window

      if not monthly:
        window_forecast.append(self.Y_forecast_df.iloc[2])   # get values 3 months ahead
      else:
        window_forecast.append(self.Y_forecast_df.iloc[0])   # get values 1 month ahead

    window_forecast = pd.DataFrame(window_forecast)
    window_errors   = window_forecast - self.df_Y.loc[window_forecast.index]

    self.window_forecast = window_forecast
    self.window_errors = window_errors

class DFM_SSM:

  '''
  This class models a DFM using factors estimated by PPCA and other endogenous regressors
  This class deals with missing data through the EM algorithem.
  '''
  def __init__(self, df_Y, factor_dict, Y_lags, start_est_date='1999-01-01', end_est_date='2020-03-01', quarterly =0 , monthly =1, daily =1):
    self.df_Y               = df_Y
    self.factor_dict        = factor_dict
    self.Y_lags             = Y_lags
    self.T                  = 25
    self.start_est_date     = start_est_date
    self.end_est_date       = end_est_date
    self.check_performance  = False
    self.verbose_likelihood = True

  def _Create_Lags(self, df, lags):
    '''
    this function takes in a pd.dataframe and returns a dataframe with the lagged data for a specified amount of lags
    as well as the original df that was passed

    INPUTS:
    df = pd.DataFrame()
    lags = integer  # specifing the number of lags

    OUTPUTS:
    df_org = pd.DataFrame() # original df passed
    df_lag = pd.DataFrame() # lags of df passed
    '''
    cols = df.shape[1]
    df_lag = np.zeros((len(df)-lags,cols*lags))

    # loop through all lags
    for lag in range(lags, 0, -1):
        df_lag[:,0+cols*(lags-lag):cols*(lags-lag+1)] = df.iloc[lag-1:(len(df)-lags+lag-1),:];

    # create copy of original df to cut first "lag" rows
    df_og = df.copy()
    df_og = df_og.iloc[lags:,:]

    # create index as dates and convert df_lag to df
    df_lag_dict = {'date': df_og.index}
    df_lag_dict = {}
    for series in df_og.columns:
        #retrieve series index
        col_index = df_og.columns.get_loc(series)

        # append and name all series with correct lags to df_lag
        for lag in range(lags):
            serie_lag = series+" lag "+str(lag+1)
            df_lag_dict[serie_lag] = df_lag[:, col_index*lags + lag]

    # create index as dates and convert df_lag to df
    df_lag = pd.DataFrame(df_lag_dict)
    df_lag.index = pd.DatetimeIndex(df_og.index)
    return df_og, df_lag

  def _Create_SSM(self, params):
    '''
    unpacks params array and creates the SSM.
    Y_t = Z x_t + e_t
    x_t = W x_t_1 + R u_t

    simply example set up with 2 factors and a global factor (4 series: 2 per factor)

    Z   = [1 0 1]  [N x m]
          [1 0 1]
          [0 1 1]
          [0 1 1]

    W   = [ 1 0 1]  [m x m]
          [ 0 1 1]
          [ 0 0 1]


    Q   = [ 1 0 0]  [m x m]
          [ 0 1 0]
          [ 0 0 1]

    R   = [ 1 0 0]  [m x m]
          [ 0 1 0]
          [ 0 0 1]

    H   = [ 1 0 0 0]  [N x N]
          [ 0 1 0 0]
          [ 0 0 1 0]
          [ 0 0 0 1]

    we need to unpack params for W, Z, R, and H. Meaning params will be an array of size (3N + 3m - 1)

    INPUT:
    params = np.array()

    OUTPUTS:
    Z = np.array()
    W = np.array()
    R = np.array()
    H = np.array()
    Q = np.array()
    '''
    factor_dict = self.factor_dict

    N = self.N
    m = self.m

    #params = np.ones(3*N + 3*m - 1)

    #for i, var_i in enumerate(args):
    # params[i] = args[var_i]

    # this is the index that we will continue to update throughout the function
    param_i = 0

    ############
    # create Z #
    ############
    Z     = np.zeros((N,m))
    Z_row = 0                               # starting row that will be updated as we unpack params

    for i, factor in enumerate(factor_dict):
      n = len(factor_dict[factor].columns)     # get number of series in a factor
      if factor == self.global_key:
        pass
      else:
        Z[Z_row:(Z_row+n), i] = params[param_i:(param_i+n)]   # unpack param values
        Z_row   += n
        param_i += n
      #update how many rows and params went through


    # append global loadings to Z
    Z[:,-1] = params[param_i:(param_i+N)]
    param_i += N

    ############
    # create W #
    ############

    # append along the diagonal first
    W = np.identity(m)
    W = W * params[param_i:(param_i+m)]
    param_i += m

    # append m-1 missing global loadings
    W[:-1,-1] =  params[param_i:(param_i+m-1)]
    param_i += m - 1

    ############
    # create R #
    ############

    R = np.identity(m) * params[param_i:(param_i+m)]
    param_i += m

    ############
    # create H #
    ############

    H = np.identity(N) * params[param_i:(param_i+N)]
    param_i += N

    ############
    # create Q #
    ############

    Q = np.identity(m)

    return Z, W, R, H, Q

  def _Kalman_Filter(self, params, states=False):
    '''
    Caluculates the log-likelihood of the states given a set of paramters.
    Y_t = Z x_t + e_t
    x_t = W x_t_1 + R u_t

    params = np.array()          # this is passed to the _Create_SSM() function to unpack into the SSM

    OUTPUTS:
    log_likelihood = flt()
    '''
    start_est_date = self.start_est_date
    end_est_date   = self.end_est_date

    factor_dict = self.factor_dict

    Z, W, R, H, Q = self._Create_SSM(params)                       # unpack params vector into SSM

    # pick the correct data to estimate the factors on
    check_performance = self.check_performance
    if check_performance:
      y = factor_dict[self.global_key].loc[start_est_date:end_est_date].values                   # y is all series in global factor   # for pre-covid estimation
    else:
      # get the most realized date for estimation
      most_realized_date = factor_dict["global"].last_valid_index()
      for serie in factor_dict["global"]:
        if factor_dict["global"][serie].last_valid_index() < most_realized_date:
          most_realized_date = factor_dict["global"][serie].last_valid_index()
      y = factor_dict[self.global_key].loc[start_est_date:most_realized_date].values                   # y is all series in global factor

    T = len(y)
    N = self.N
    m = self.m

    # initialize with first iteration of the Kalman filter
    log_likelihood = -(T*N/2)*np.log(2*np.pi)
    mu_0   = np.ones((m,1))
    x_hat  = np.zeros((T, m))
    P      = np.zeros((T, m, m))
    e      = y[0] - (Z @ mu_0).T
    SS     = Z @ P[0] @ Z.T + H @ H.T
    SS_inv = np.linalg.inv(SS)
    log_likelihood = - (1/2)*np.log(np.linalg.det(SS)) - (1/2)* e @ SS_inv @ e.T;

    # set priors for filter
    x_hat[0] = 0
    P[0]     = np.identity(m)

    # Run the Kalman filter
    try:
      for t in range(1, T):
          # Prediction step
          x_hat[t] = W @ x_hat[t-1]
          P[t]     = W @ P[t-1] @ W.T + R @ R.T
          SS       = Z @ P[t] @ Z.T + H @ H.T
          SS_inv   = np.linalg.inv(SS)

          # Update Step
          e        = y[t] - (Z @ x_hat[t])
          x_hat[t] = x_hat[t] + P[t] @ Z.T @ SS_inv @ e
          P[t] = P[t] - P[t] @ Z.T @ SS_inv @ Z @ P[t]

          # update likelihood
          log_likelihood = log_likelihood - (1/2)*np.log(np.linalg.det(SS)) - (1/2)* e @ SS_inv @ e.T;
    except:
      print("Current parameters resulted in SingularMatrix error -- pass to next iteration")

    if states:
      factor_df = pd.DataFrame()
      for i, factor in enumerate(factor_dict):
        factor_df[factor] = x_hat[:,i]
      if check_performance:
        factor_df.index = factor_dict[self.global_key].loc[start_est_date:end_est_date].index     # for pre-covid estimation
      else:
        factor_df.index = factor_dict[self.global_key].loc[start_est_date:most_realized_date].index
      return factor_df

    else:
      verbose = self.verbose_likelihood
      if verbose: 
          print("log-likelihood with current params: " + str(log_likelihood[-1][0]))
      return log_likelihood[-1][0]

  def _Get_States(self):
    '''
    this function optimizes the log-likelihood from the Kalman filter given the
    array of parameters. The function then returns the states with the optimal
    parameters.
    '''
    N = self.N
    m = self.m

    bnds_tuple   = (-0.99, 0.99)
    bnds_tuple_R = (-1, 1)                                               # place tighter bounds on covariance matric R
    bnds         = ((bnds_tuple,) * (3*N+2*m-1))
    bnds         = bnds + ((bnds_tuple_R,) * (m))



    x0 = np.ones(3*self.N+3*self.m-1)                                    # initial guess to ones
    result = minimize(self._Kalman_Filter, x0, bounds=bnds, tol=1e-7)
    self.SSM_params = result.x
    self.factor_df  = self._Kalman_Filter(self.SSM_params, states=True)  # rerun the Kalman filter with the optimal parameters

  def _Est_VAR(self, df_y, lags, df_exog=None, constant=1):
    '''
    INPUT:
    df_y = pd.DataFrame() # dataframe of dependent variables -- df_y must have a column name
    lags = integer        # corresponds to number of lags for df_y
    df_exog = pd.DataFrame() # dataframe of exogeneous variables (defaults to None)
    constant = bool       # specifies whether to include intercept in regression (defaults to 1)

    OUTPUT:
    results = dictionary  # within the dictionary there are many results linked to estimating a VAR
    results['x'] = np.array()                 # regressors
    results['y'] = np.array()                 # dependent var
    results['nvars'] = integer                # number of variables
    results['nobs'] = integer                 # number of observations
    results['ncoefs'] = integer               # number of coefficients
    results['beta'] = np.array()              # array of estimated betas NOTE: the final beta corresponds to the intercept
    results['sigma'] = np.array()             # covariance matrix
    results['vcovbeta'] = np.array()          # coefficient covariance matrix
    results['stdbeta'] = np.array()           # std of beta
    results['companion_mat'] = np.array()     # companion matrix
    results['aic'] = integer
    results['bic'] = integer
    results['hq'] = hq
    results['Yfit'] = np.array()              # fitted Y values
    results['SSR'] = integer                  # sum of squared residuals
    results['lags'] = integer                 # number of lags
    '''

    if lags == 0 and constant == 0:
        raise ValueError('WARNING: you are trying to estimate a VAR without a constant or lags')
    if lags > df_y.shape[0]:
        raise ValueError('WARNING: you must have more observations than lags')
    #if df_y.shape[0] < df_y.shape[1]:
      # print(df_y.shape[0])
      #print(df_y.shape[1])
        #raise ValueError('WARNING: You have more variables than observations or you have entered the data incorrectly.')

    # set estimation period
    start_est_date = self.start_est_date
    end_est_date   = self.end_est_date

    results = {}

    # initialize helper variables
    nvars = df_y.shape[1]
    nobs = df_y.shape[0] - lags
    ncoefs = nvars * lags + len(df_exog.columns) + constant
    ibeg = lags + 1
    iend = df_y.shape[0]

    # initialize dependent variable and its lags
    z, xz = self._Create_Lags(df_y, lags)

    # restrict sample to precovid estimation period
    z  = z.loc[start_est_date:end_est_date]
    xz = xz.loc[start_est_date:end_est_date]

    if df_exog is not None:
        xz = pd.concat([xz, df_exog.loc[xz.index]], axis = 1)

    # add a constant term
    if constant:
        df_ones = pd.DataFrame(np.ones(xz.shape[0]))
        df_ones.set_index(xz.index, inplace=True)
        xz = pd.concat([xz, df_ones], axis = 1)
        #xz = pd.concat([xz, pd.DataFrame(np.ones(xz.shape[0]))], axis = 1)
        #xz = np.concatenate((xz, np.ones(xz.shape[0])), axis=1)

    beta = np.linalg.lstsq(xz, z, rcond=None)[0]
    Yfit = np.dot(xz, beta)
    uz = z - Yfit
    SSR = np.sum(uz**2)
    sigma = SSR / (len(uz) - lags * nvars + 1)
    vcovbeta = np.kron(sigma, np.linalg.inv(np.dot(xz.T, xz)))
    #SX = np.sqrt(np.diag(vcovbeta)).reshape(nvars, nvars)
    SX = 1
    companion_mat = np.vstack((beta[:lags * nvars, :].T, np.eye((lags - 1) * nvars, lags * nvars)))

    ldet = np.log(np.linalg.det(np.cov(uz)))
    nbc = beta.size
    aic = ldet + 2 * nbc / nobs
    bic = ldet + nbc * np.log(nobs) / nobs
    hq = ldet + 2 * nbc * np.log(np.log(nobs)) / nobs

    results['x'] = xz
    results['y'] = z
    results['nvars'] = nvars
    results['nobs'] = nobs
    results['ncoefs'] = ncoefs
    results['beta'] = beta
    results['sigma'] = sigma
    results['vcovbeta'] = vcovbeta
    results['stdbeta'] = SX
    results['companion_mat'] = companion_mat
    results['aic'] = aic
    results['bic'] = bic
    results['hq'] = hq
    results['Yfit'] = Yfit
    results['SSR'] = SSR
    results['lags'] = lags
    return results

  def _Get_Estimates(self, check_performance=False):
    '''
    call multiple functions to get the parameters of the VAR model
    '''
    self.global_key = [*self.factor_dict.keys()][-1]
    self.N = len(self.factor_dict[self.global_key].columns)
    self.m = len(self.factor_dict)

    self._Get_States()              # update the states and get self.factor_df

    model_params      = self._Est_VAR(self.df_Y, self.Y_lags, self.factor_df)
    self.model_params = model_params

    return model_params

  def _Get_State_Forecast(self, T):
    '''
    get the factor forcast up to time T

    INPUTS:
    state_df      = pd.DataFrame()
    T              = int              # horizon of what you would like to forcast to
    forecast_date  = str()            # start date of forecast defaults to current date

    OUTPUTS:
    forecast_state_df = pd.DataFrame()  # size (T, N_factors)
    '''
    check_performance = self.check_performance
    start_est_date    = self.start_est_date

    # choose correct dates to forecast from
    if check_performance:
      end_est_date    = self.end_est_date
    else:
      end_est_date    = self.df_Y.last_valid_index()
    factor_df         = self.factor_df

    # restrict factor forecast to subsample
    state_df = factor_df.loc[start_est_date:end_est_date]

    # unpack SSM from optimal params:
    Z, W, R, H, Q = self._Create_SSM(self.SSM_params)

    # we just want to use the params of the SSM model that we estimated over a certain period to get our predict values of states
    x_hat    = np.zeros((T, self.m))
    x_hat[0] = factor_df.loc[end_est_date]

    for t in range(1, T):
      x_hat[t] = W @ x_hat[t-1]

    factor_forecast_dict = {}
    for i, factor in enumerate(factor_df):
      factor_forecast_dict[factor] = x_hat[:, i]

    # get forecast dates
    forecast_dates     = pd.date_range(start=end_est_date, periods=T+1, freq='MS')

    # create dataframe
    factor_forecast_df = pd.DataFrame(factor_forecast_dict)
    factor_forecast_df.set_index(forecast_dates[1:], inplace=True)

    # concate dataframe so that we can later choose the correct data from forecast dates
    factor_forecast_df = pd.concat([factor_df, factor_forecast_df])

    self.factor_forecast_df = factor_forecast_df

    return factor_forecast_df

  def _Create_Forecast(self, forecast_date=None):
    '''
    Note: the model_params['beta'] arrary is a (N_vars * N_lags + N_factors + 1 ,N_vars) array
    for example with N_vars = 3, N_lags = 3, N_factors = 3

    |   Var1   |   Var1   |   Var1   |
    ----------------------------------
    |  Var1L1  |  Var1L1  |  Var1L1  |
    |  Var1L2  |  Var1L2  |  Var1L2  |
    |  Var1L3  |  Var1L3  |  Var1L3  |
    |  Var2L1  |  Var2L1  |  Var1L1  |
    |  Var2L2  |  Var2L2  |  Var2L2  |
    |  Var2L3  |  Var2L3  |  Var2L3  |
    |  Var3L1  |  Var3L1  |  Var3L1  |
    |  Var3L2  |  Var3L2  |  Var3L2  |
    |  Var3L3  |  Var3L3  |  Var3L3  |
    |  factor1 |  factor1 |  factor1 |
    |  factor2 |  factor2 |  factor2 |
    |  factor3 |  factor3 |  factor3 |
    |  cons.   |  cons.   |  cons.   |

    INPUTS:
    self.Y_df          = pd.DataFrame()    # contains historical data for the dependent variables
    self.factor_df     = pd.DataFrame()    # contains historical data for the factors
    self.Y_lags        = int               # number of lags of dependent variable
    self.T             = int               # forecast horizon
    self.betas         = np.array()        # of the form above
    self.forecast_date = str()             # if not specified defaults to current date

    OUTPUTS:
    Y_forecast_df = pd.DataFrame()  # (T, N_var) forecast for all dependent variables
    '''
    Y_df      = self.df_Y
    factor_df = self.factor_df
    Y_lags    = self.Y_lags
    beta      = self.model_params["beta"]
    T         = self.T

    N_vars    = len(Y_df.columns)
    N_lags    = 3
    N_factors = len(factor_df.columns)

    # restrict Y_df to have data up to forecast_date
    if forecast_date != None:
      Y_df    = Y_df.loc[:forecast_date]

    # get factor forecast up to time T and get forecast dates
    forecast_dates      = pd.date_range(start=Y_df.index[-1], periods=T+1, freq='MS')
    self.forecast_dates = forecast_dates
    factor_forecast_df  = self._Get_State_Forecast(T)      # get factor forecast
    factor_forecast_df  = factor_forecast_df.loc[forecast_dates]


    # get each forecast t+1 by creating array below and multiplying by the betas array
    # [Var1L1, Var1L2, Var1L3, Var2L1, Var2L2, Var2L3, Var3L1, Var3L2, Var3L4, factor1, factor2, factor3, 1]

    for t in range(T):
      # initialize the array
      realized_array = np.zeros([N_vars * N_lags + N_factors + 1,1])
      for i, y in enumerate(Y_df.columns):
        realized_array[i*N_lags:(i+1)*N_lags] = np.array(Y_df[y].iloc[-N_lags:]).reshape(-1,1)

      # add the factor to the realized array
      realized_array[(N_vars)*N_lags:(N_vars)*N_lags + N_factors] = np.array(factor_forecast_df.iloc[t,:]).T.reshape(-1,1)

      # add the constant to the realized array
      realized_array[-1] = 1

      # get forecast for the current period
      forecast_array   = np.dot(realized_array.T, beta)
      ts = pd.to_datetime(forecast_dates[t+1], format="%Y-%m-%d %H:%M:%S.%f")     # get timestamp of t+i
      new_row = pd.DataFrame(forecast_array, columns=Y_df.columns, index=[ts])
      Y_df = pd.concat([Y_df, pd.DataFrame(new_row)], ignore_index=False)         # concat with previous row and iterate

    self.Y_forecast_df = Y_df.iloc[-T:]
    self.factor_forecast_df = factor_forecast_df

    return Y_df.iloc[-T:]

  def _Check_Model_Performance(self, window_start="01-01-2010", window_end="01-01-2020", monthly=False):
    '''
    check the 1 quarter ahead out of sample forecast of model using rolling window estimation

    INPUTS:
    window_start = str()         # date when we start the rolling window estimation
    window_end   = str()         # date when we end the rolling window estimation
    monthly      = bool          # True if want monthly forecasts otherwise defaults to quarterly

    OUTPUTS:
    self.window_forecast  = pd.DataFrame()       # 3 or 1 month ahead forecasts
    self.window_errors    = pd.DataFrame()       # 3 or 1 month ahead forecast errors
    '''
    self.check_performance  = True
    self.verbose_likelihood = False

    window_forecast = []
    window_dates = pd.date_range(start=window_start, end=window_end, freq='MS')

    t_increment = 1   # how much we increase rolling window by

    if not monthly:
      window_dates = pd.date_range(start=window_start, end=window_end, freq='3MS')
      t_increment  = 3

    print("checking model performance from: " + str(window_dates[0]) + " to " + str(window_dates[-1]))
    for date in window_dates:
      self.start_est_date = str((datetime.strptime(self.start_est_date, '%Y-%m-%d').date() + pd.DateOffset(months=t_increment)).date())
      self.end_est_date   = date                   # change estimation date over the window
      self._Get_Estimates(check_performance=True)  # update parameters
      self._Create_Forecast(forecast_date=date)    # get forecast over window
      print("estimation from " + str(self.start_est_date) + " to " + str(self.end_est_date) +  " complete.")

      if not monthly:
        window_forecast.append(self.Y_forecast_df.iloc[2])   # get values 3 months ahead
      else:
        window_forecast.append(self.Y_forecast_df.iloc[0])   # get values 1 month ahead

    window_forecast = pd.DataFrame(window_forecast)
    window_errors   = window_forecast - self.df_Y.loc[window_forecast.index]

    self.window_forecast = window_forecast
    self.window_errors = window_errors
    self.check_performance  = False
    self.verbose_likelihood = True