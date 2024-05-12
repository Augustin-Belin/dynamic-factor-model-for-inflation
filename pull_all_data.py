from fredapi import Fred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pprint import pprint
from pybea.client import BureauEconomicAnalysisClient
from datetime import date
from IPython.display import HTML
from yahoo_fin.stock_info import get_data

def _Pull_Fred_Data(df_data_id_codes, start_date, verbose=True, quarterly=1, monthly=1, weekly=1, daily=1, factor=1):
    '''
    function pulls data from a .csv file with specified format:
    df_data_id_codes.columns = ["id", "desc", "factor", "units", "frequency"]

    INPUTS:
    df_data_id_codes = pd.DataFrame
    start_date = str                  # specify date you would like to start pulling data from
    quarterly  = bool                 # 1 -> pull quarterly data, 0 -> do not pull quarterly data
    montlhly   = bool                 # 1 -> pull montlhly data, 0 -> do not pull montlhly data
    daily      = bool                 # 1 -> pull daily data, 0 -> do not pull daily data
    factor     = bool                 # 1 -> pull factor data, 0 -> pull pce data

    OUTPUTS:
    factor_1   = pd.DataFrame()
    factor_2   = pd.DataFrame()
    factor_3   = pd.DataFrame()
    factor_3   = pd.DataFrame()
    '''
    # NOTE
    # --------------------------------------------------------------------------
    # below we provide 3 api keys. Sometimes the max pull amount is exceeded
    # and the api will no longer work. In that case switch the api.
    # --------------------------------------------------------------------------
    #fred = Fred(api_key='63b3cd25139580c1b976833da5a1ee58')
    #fred = Fred(api_key='b1faaf3fb72665c60817a0a2a8bdf9f6')
    fred = Fred(api_key='9b2b344af693b65a50a00f459c585bb6')

    # initialize
    factor_names = list(set(df_data_id_codes["factor"]))
    factor_dict  = {}

    for name in factor_names:
      factor_dict[name] = pd.DataFrame()

    # initialize y dataframe
    df_y       = pd.DataFrame()

    # iterate through the codes provided in df_data_id_codes
    for i,row in df_data_id_codes.iterrows():
      try:
        # only append series with specified frequencies
        if row["frequency"] == "daily" and daily == 0:
          print("passed daily series:"," series", "[id: ", row["id"], "]")
          continue
        if row["frequency"] == "weekly" and weekly == 0:
          print("passed weekly series:"," series", "[id: ", row["id"], "]")
          continue
        if row["frequency"] == "monthly" and monthly == 0:
          print("passed monthly series"," series", "[id: ", row["id"], "]")
          continue
        if row["frequency"] == "quarterly" and quarterly == 0:
          print("passed quarterly series:"," series", "[id: ", row["id"], "]")
          continue

        series = fred.get_series(row["id"])        # pull serie value
        info   = fred.get_series_info(row["id"])   # pull serie info

        # some of the data is daily, as such take the mean over the month
        if row["frequency"] == "daily":
          series      = series.resample('M').mean()
          series.index = series.index - pd.offsets.MonthBegin(1) # change index to be the first of each month

        # some of the data is weekly, as such take the mean over the month
        if row["frequency"] == "weekly":
          series      = series.resample('M').mean()
          series.index = series.index - pd.offsets.MonthBegin(1) # change index to be the first of each month

        # check to see if series is already stationary
        if is_stationary(series.dropna()) < 0.05: stationary_str = "series is stationary"

        # if it is not we take the percent change and check for stationarity again
        else:
          series = series.pct_change()
          if is_stationary(series.dropna()) < 0.05: stationary_str = "series is stationary after transformation"
          else:
            stationary_str = " "

        # append series to correct factors based on "factor" column
        factor_dict[row["factor"]][row["desc"]] = series[start_date:]

        # append to y df if needed
        if row["factor"] == "y":
          df_y[row["desc"]] = series[start_date:]

        if verbose:
          print("succesfully pulled: ", i+1, " out of ", len(df_data_id_codes), " series", "[id: ", row["id"], "]", stationary_str)

      # sometimes the API key reaches max number of pulls, in that case change keys
      except:
        fred = Fred(api_key='b1faaf3fb72665c60817a0a2a8bdf9f6')
        print("API KEY CHANGED -- max number of requests exceeded")

    if factor == 1:
      return factor_dict
    else:
      return df_y
    

def is_stationary(series):
  '''
  takes in a series of the form of df

  INPUT:
  series = pd.DataFrame()

  OUTPUT:
  adf_test[1] = integer    # p_value from ad fuller test for stationarity
  '''
  #
  adf_test = adfuller(series, regression = 'ct')
  # the second value of the adf_test represents the p-value of the test
  return adf_test[1]

def _Clean_BEA_Data(pce_dict, verbose = True):
  # initialize 3 sectors
  goods_categories_list    = ["Motor vehicles and parts", "Furnishings and durable household equipment", "Recreational goods and vehicles", "Other durable goods", "Clothing and footwear", "Other nondurable goods"]
  services_categories_list = ["Health care", "Transportation services", "Recreation services", "Food services and accommodations", "Financial services and insurance", "Other services", "Final consumption expenditures of nonprofit institutions serving households (NPISHs)"]
  housing_categories_list  = ["Housing"]

  goods_dict    = {}
  services_dict = {}
  housing_dict  = {}

  dates         = []

  entry_count   = 1

  # check to see which table we are in to properly switch string to float
  if pce_dict[0]["TableName"] == "T20805":
    replace_str = ","
  if pce_dict[0]["TableName"] == "T20804":
    replace_str = "."

  # loop through
  for entry in pce_dict:

    # check to see if entry is in goods
    if entry["LineDescription"] in goods_categories_list:
      if entry["LineDescription"] not in goods_dict:
          goods_dict[entry["LineDescription"]] = []
      goods_dict[entry["LineDescription"]].append(float(entry["DataValue"].replace(replace_str, "")))

    # check to see if entry is in services
    if entry["LineDescription"] in services_categories_list:
      if entry["LineDescription"] not in services_dict:
          services_dict[entry["LineDescription"]] = []
      services_dict[entry["LineDescription"]].append(float(entry["DataValue"].replace(replace_str, "")))

    # check to see if entry is in housing
    if entry["LineDescription"] in housing_categories_list:
      if entry["LineDescription"] not in housing_dict:
          housing_dict[entry["LineDescription"]] = []
      housing_dict[entry["LineDescription"]].append(float(entry["DataValue"].replace(replace_str, "")))

    # add date
    # first have to transform the date format from "YYYYMmm" to "mmddYYYY"
    date = entry["TimePeriod"].replace('M', '')[4:7] + "-" + "01" + "-" + entry["TimePeriod"].replace('M', '')[0:4]

    if date not in dates:
      dates.append(date)

    if verbose:
      print("succesfully pulled entry: ", entry_count, " out of ", len(pce_dict))
      entry_count += 1

  # append dictionaries to dataframes and changes to datatime index
  goods_df = pd.DataFrame(goods_dict)
  goods_df.index = pd.DatetimeIndex(dates)

  services_df = pd.DataFrame(services_dict)
  services_df.index = pd.DatetimeIndex(dates)

  housing_df = pd.DataFrame(housing_dict)
  housing_df.index = pd.DatetimeIndex(dates)

  return goods_df, services_df, housing_df

def _Pull_BEA_Data(start_date="1970"):
    '''
    pull BEA id codes and description of all sectors of PCE
    #########################################################
    format of json is as follows
    pce_index_json["BEAAPI"]["Results"]["Data"]
    and within this dictionary the options are
    |'TableName'|'SeriesCode'|'LineNumber'|'LineDescription'|'TimePeriod'|'METRIC_NAME'
    |'CL_UNIT'|'UNIT_MULT'|'DataValue'|'NoteRef'
    ####################################################

    INPUTS:
    start_date = str(YYYY)

    OUTPUTS:
    agg_goods_inf = pd.DataFrame()   # aggregate inflation for goods, services, and housing 
    agg_pce_inf   = pd.DataFrame()   # aggregate inflation PCE
    agg_shares    = pd.DataFrame()   # aggregate shares goods, services, and housing 
    agg_exp       = pd.DataFrame()   # aggregate expenditures goods, services, and housing 
    '''

    bea_api = "E8646E50-F4E4-458F-9194-2CC374585B65"

    # initialize API client
    bea_client = BureauEconomicAnalysisClient(api_key = bea_api)
    dataset_list = bea_client.get_dataset_list()
    parameter_set_list = bea_client.get_parameters_list(dataset_name = 'NIPA')

    # choose years to pull from
    start_date = "1970"
    end_date   = date.today()
    end_date   = end_date.strftime("%Y")
    temp_years = list(range(int(start_date), int(end_date)+1))
    years = [str(e) for e in temp_years ]

    # get data for expenditures
    pce_expenditures_json = bea_client.national_income_and_product_accounts_detail(
        table_name = "T20805",                                                      # Table 2.8.5. Personal Consumption Expenditures by Major Type of Product, Monthly
        frequency  = ["M"],
        year       = years
    )
    pce_expenditures_dict = pce_expenditures_json["BEAAPI"]["Results"]["Data"]

    # get data for indexes
    pce_index_json = bea_client.national_income_and_product_accounts_detail(
        table_name = "T20804",                                                      # Table 2.8.4. Price Indexes for Personal Consumption Expenditures by Major Type of Product, Monthly
        frequency  = ["M"],
        year       = years
    )
    pce_index_dict = pce_index_json["BEAAPI"]["Results"]["Data"]

    # clean data from JSON format
    goods_exp, services_exp, housing_exp = _Clean_BEA_Data(pce_expenditures_dict, verbose = False)
    goods_ind, services_ind, housing_ind = _Clean_BEA_Data(pce_index_dict, verbose = False)

    # get shares and inflation dataframes
    goods_inf, services_inf, housing_inf        = goods_ind.pct_change(), services_ind.pct_change(), housing_ind.pct_change()
    goods_share, services_share, housing_share  = goods_exp.div(goods_exp.sum(axis=1), axis=0), services_exp.div(services_exp.sum(axis=1), axis=0), housing_exp.div(housing_exp.sum(axis=1), axis=0)

    # get aggregate inflation for each aggregate sector
    agg_goods_inf, agg_services_inf, agg_housing_inf = (goods_inf*goods_share).sum(axis=1), (services_inf*services_share).sum(axis=1), (housing_inf*housing_share).sum(axis=1)
    agg_sectors_inf = pd.DataFrame({"goods":agg_goods_inf, "services":agg_services_inf, "housing":agg_housing_inf})

    # get core pce inflation
    agg_exp     = pd.DataFrame({"goods":goods_exp.sum(axis=1), "services":services_exp.sum(axis=1), "housing":housing_exp.sum(axis=1)})
    agg_shares  = agg_exp.div(agg_exp.sum(axis=1), axis=0)
    agg_pce_inf =  pd.DataFrame({"pce":(agg_sectors_inf*agg_shares).sum(axis=1)})

    return agg_sectors_inf, agg_pce_inf, agg_shares, agg_exp 

def _Pull_GSCPI_Data(start_date):
  '''
  pull GSCPI from NY FED website 

  OUTPUTS: 
  GSCPI = pd.DataFrame()
  '''
  url_GSCPI   = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
  GSCPI_temp  = pd.read_excel(url_GSCPI, sheet_name="GSCPI Monthly Data")
  GSCPI       = pd.DataFrame(GSCPI_temp["GSCPI"].dropna())
  GSCPI_dates = pd.to_datetime(GSCPI_temp["Date"].dropna()) - pd.offsets.MonthBegin(1)  # set the day to the 1st of the month 
  GSCPI.set_index(GSCPI_dates, inplace=True)
  GSCPI.index.name = None

  return GSCPI.loc[start_date:]

def _Pull_Yahoo_Data(tickers, start_date):
  '''
  pull specified tickers from YaHoo Finance 

  INPUTS:
  tickers         = list
  start_date_pull = str

  OUTPUT:
  index_df        = pd.DataFrame()
  '''
  index_df    = pd.DataFrame()

  # pull data from yahoo API 
  for ticker in tickers:
    serie       = get_data(ticker, index_as_date = True, interval="1d")
    serie_cls       = serie["close"]
    serie_cls       = serie_cls.resample('M').mean()             # change to monthly data 
    serie_cls.index = serie_cls.index - pd.offsets.MonthBegin(1) # change index to be the first of each month
    serie_pct       = serie_cls.pct_change()
    
    index_df[ticker] = serie_pct.loc[start_date:]

  return index_df

def _Get_Factor_Table(data_id_codes):
  '''
  makes a pretty table to see which series belong to each factor 

  INPUTS:
  data_id_codes = pd.DataFrame()

  OUTPUT:
  None
  '''
  # get all unique factor names 
  factor_names = list(set(data_id_codes["factor"]))
  factor_dict  = {}

  for factor in factor_names:
    factor_dict[factor] = []

  # append each serie to factor_dict and add "X" if they are in the factor 
  for i,row in data_id_codes.iterrows():
    for factor in factor_dict:
      if row["factor"] == factor:
        factor_dict[factor].append("X")
      else:
        factor_dict[factor].append(" ")

  factor_dict["series"]    = data_id_codes["desc"]
  factor_dict["frequency"] = data_id_codes["frequency"]


  # reorg the column names and create dataframe that will be displayed
  col_names   = ["series", "frequency"]
  col_names.extend(factor_names)
  factor_desc = pd.DataFrame(factor_dict)
  factor_desc = factor_desc.loc[:, col_names]
  factor_desc = factor_desc.style.set_properties(**{'text-align': 'left'})

  display(HTML(factor_desc.to_html(index=False)))

  return factor_desc
