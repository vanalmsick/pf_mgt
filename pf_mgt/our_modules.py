# Here all general code modules are collected to reuse them
import datetime
import datetime as dt
import math
import os
import pathlib
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypfopt as pfopt
import seaborn as sns
from packaging import version
from tabulate import tabulate
from scipy.stats import norm


def get_data_nice_looking(fname, sname, sheet_props=None, first_row=0, last_row=None, first_col=0, last_col=None, use_cache=True):
    """
    Import data from Excel and make it nice looking. This function is also able to catch the data for faster imports.

    Args:
        fname (str): file name and path
        sname (str): sheet name to import
        sheet_props (dict): arguments for importing which are forwarded to pd.read_excel e.g. {'index_col': 0, 'header': 1, 'parse_dates':'index'}
        first_row (int): change first row to row number
        last_row (int): change last row to row number
        first_col (int): change first column to column number
        last_col (int): change last column to column number
        use_cache (bool): catch import for faster execution

    Returns:
        DataFrame of the respective sheet of the respective file

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """

    if sheet_props == None or sheet_props == 'default':
        sheet_props = {'index_col': 0, 'header': 0}
    elif type(sheet_props) == dict:
        pass
    else:
        raise Exception('Please specify a dictionary or "default" as sheet_props.')


    if 'parse_dates' not in sheet_props:
        sheet_props['parse_dates'] = True
    elif type(sheet_props['parse_dates'])==str:
        sheet_props['parse_dates'] = [sheet_props['parse_dates']]

    if type(sheet_props['parse_dates'])==list:
        if 'index' in sheet_props['parse_dates']:
            if len(sheet_props['parse_dates']) == 1:
                sheet_props['parse_dates'] = True
                current_sheet_props = ['index']
            else:
                sheet_props.remove('index')
                current_sheet_props = sheet_props['parse_dates']
        else:
            current_sheet_props = sheet_props['parse_dates']
    else:
        current_sheet_props = None


    def new_data(path, sname, current_sheet_props, save):
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sname, **current_sheet_props)
        if save:
            df_dict = sheet_props
            df_dict['DataFrame'] = df
            pathlib.Path('tmp/').mkdir(parents=True, exist_ok=True)
            pickle.dump(df_dict, open('tmp/' + sname + '.df.pickle', "wb+"))
        return df

    if use_cache == True and os.path.isfile('tmp/' + sname + '.df.pickle'):
        df_dict = pickle.load(open('tmp/' + sname + '.df.pickle', 'rb'))
        df = df_dict['DataFrame']
        del df_dict['DataFrame']
        if type(sheet_props['parse_dates'])==list:
            if 'index' in sheet_props['parse_dates']:
                sheet_props['parse_dates'] = sheet_props['parse_dates'].remove('index')
        if df_dict != sheet_props:
            df = new_data(fname, sname, sheet_props, save=True)
    elif use_cache == True:
        df = new_data(fname, sname, sheet_props, save=True)
    else:
        df = new_data(fname, sname, sheet_props, save=False)

    if last_row == None:
        df = df.iloc[first_row:]
    else:
        df = df.iloc[first_row:last_row]
    if last_col == None:
        df = df[list(df.columns)[first_col:]]
    else:
        df = df[list(df.columns)[first_col:last_col]]

    if type(current_sheet_props)==list:
        for i in current_sheet_props:
            if i == 'index':
                df.index = pd.to_datetime(df.index)
            else:
                df[i] = pd.to_datetime(df[i])

    return df


def DF_column_currency_conversion(df, columns, FX_rate_df):
    """
    This function converts the specified columns of a DatafRame with an FX rate DateFrame to one currency.

    Args:
        df (pd.DataFrame): entire DataFrame
        columns (str/list): name or list of columns to transform with the FX rate
        FX_rate_df (pd.DataFrame): FX rate 1D DataFrame to convert specified columns with

    Returns:
        DataFrame with the columns specified converted to other currency

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    if type(columns)!=list:
        columns = list(columns)
    for column in columns:
        df[column] = df[column] * FX_rate_df
    return df



def DF_column_bond_conversion(df, column, rebalancing='m', holding_days=91, daycount=360, output='intermediate steps'):
    '''
    This function adds two columns to the input dataframe DF:
    TBILL_Prices:   This column contains the prices of the TBILLs in the portfolio
    TBILL_Returns:  This column contains the returns of the TBILLs in the portfolio
    '''
    
    df_old = df.copy()
    df_old['RAW T-rate'] = df_old[column]
    df_old['today'] = df_old.index.to_pydatetime()
    df_new = df_old.copy()
    df_new['rebalance'] = np.nan

    
    new_today = df_new['today'].iloc[0]
    last_date = new_today - dt.timedelta(days=1)
    tmp_df = pd.DataFrame(columns=['date','rebalance'])
    for index, row in df_old.iterrows():
        if rebalancing == 'm':
            if last_date.month != index.month:
                new_today = row['today'].to_pydatetime()
                last_date = row['today']
            df_new.at[index, 'rebalance'] = new_today
        else:
            raise Exception('Rebalancing not jet coded.')
    
    df_new['today'] = df_new['today'].astype('datetime64[ns]')
    df_new['rebalance'] = df_new['rebalance'].astype('datetime64[ns]')
    df_new['remaining_days'] = holding_days - (df_new['today'] - df_new['rebalance']).dt.days
    
    df_new['T-Bill-Price']= 100/(1+df_new[column]/100*df_new['remaining_days']/daycount)
    df_new['T return'] = df_new['T-Bill-Price'].pct_change()

    i = 0
    for idx, row in df_new.iterrows():
        if row['remaining_days'] == holding_days and i != 0:
            new_rem_days = df_new.iloc[i-1]['remaining_days'] - 1
            new_price = 100/(1+row[column]/100*new_rem_days/daycount)
            new_return = new_price / df_new.at[last_idx, 'T-Bill-Price']  - 1
            df_new.at[idx, 'T return'] = new_return
        i += 1
        last_idx = idx

    df_new[column] = (1 + df_new['T return']).cumprod()
    df_new.at[df_new.index.min(), column] = 1

    if output!='intermediate steps':
        df_new.drop(columns=['RAW T-rate', 'today', 'rebalance', 'remaining_days', 'T-Bill-Price', 'T return'], inplace=True)

    # Somehow something gets messed up with the index so redo it - not elegant but works
    df_new.index = pd.to_datetime(df_new.index.strftime('%Y-%m-%d'), format='%Y-%m-%d')
    return df_new



def PF_weights(returns_DF, output, min_weight_constraint=False, show_print=False):
    """
    Get (optimal) portfolio weights given the specified method.

    Args:
        returns_DF (pd.DataFrame): DataFrame of stock returns
        output (str): specify method of weights (options: 'EF'/'MSR', 'MVol', '1/N', 'Inv')
        min_weight_constraint (float): minimum weights constraint

    Returns:
        dictionary of optimal portfolio weights

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    def inv_weigts(vola_list):
        """
        Helper function.
        """
        vola_list = np.asarray(vola_list)
        inv_vola = 1 / vola_list
        inv_vola = inv_vola / sum(inv_vola)
        return inv_vola

    def try_optimization(mu, CovS, opt_type, optimizer='CVXOPT', num_trys=3, min_weight_constraint=False, show_print=show_print):
        """
        Helper function.
        """
        if opt_type in ['MSR', 'EF']:
            for j in list(range(num_trys)):
                try:
                    # Try different optimizers
                    MSR = pfopt.EfficientFrontier(mu, CovS, solver=optimizer)  # setup
                    if min_weight_constraint != False:
                        MSR.add_constraint(lambda x: x >= min_weight_constraint)  # every stock min 1%
                    MSR.max_sharpe()  # method: Maximum Sharpe-Ratio
                    weights_EF = weights_MSR = dict(zip(MSR.tickers, MSR.weights))
                    if show_print:
                        print("Try No {} with '{}' solver was successful to solve the EF optimization.".format(str(j+1), optimizer))
                    return [0, 'Success', weights_EF]
                except:
                    if show_print:
                        print("Try No {} with '{}' solver was NOT successful to solve the EF optimization.".format(str(j+1), optimizer))
                    pass
            if show_print:
                print("No solution found in {} tries with '{}' optimizer.".format(str(j+1), optimizer))
            return [1, 'Error', 'Error. No solution found.']
        elif opt_type in ['MVol', 'MinV', 'MinVola']:
            for j in list(range(num_trys)):
                try:
                    # Try different optimizers
                    MVol = pfopt.EfficientFrontier(mu, CovS, solver=optimizer)  # setup
                    if min_weight_constraint != False:
                        MVol.add_constraint(lambda x: x >= min_weight_constraint)  # every stock min 1%
                    MVol.min_volatility()  # method: Min Volatility
                    weights_MVol = dict(zip(MVol.tickers, MVol.weights))
                    if show_print:
                        print("Try No {} with '{}' solver was successful to solve the Min Vol optimization.".format(str(j+1), optimizer))
                    return [0, 'Success', weights_MVol]
                except:
                    if show_print:
                        print("Try No {} with '{}' solver was NOT successful to solve the Min Vol optimization.".format(str(j+1), optimizer))
                    pass
            if show_print:
                print("No solution found in {} tries with '{}' optimizer.".format(str(j+1), optimizer))
            return [1, 'Error', 'Error. No solution found.']



    if output in ['MSR', 'EF', 'MVol', 'MinV', 'MinVola', 'InvVar', 'IVar', 'InvVol', 'IVol', 'InvVola', 'IVola', 'IStd', 'InvStd']:
        mu = pfopt.expected_returns.mean_historical_return(returns_DF, returns_data=True, compounding=False, frequency=252)  # False because in portfolio management assignment task "expected return MU as arithmetic mean"
        #CovS = fix_nonpositive_semidefinite(pfopt.risk_models.sample_cov(returns_DF, returns_data=True, frequency=252), fix_method="spectral")
        CovS = pfopt.risk_models.sample_cov(returns_DF, returns_data=True, frequency=252) ##pypfopt.risk_models.CovarianceShrinkage    pfopt.risk_models.sample_cov
        solvers = ['CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCS']

        if output in ['MSR', 'EF']:
            for solver in solvers:
                weights_MSR = try_optimization(mu, CovS, opt_type='MSR', optimizer=solver, num_trys=20,
                                                min_weight_constraint=min_weight_constraint)
                if weights_MSR[0] == 0:
                    break
            return weights_MSR[2]

        elif output in ['MVol', 'MinV', 'MinVola']:
            for solver in solvers:
                weights_MVol = try_optimization(mu, CovS, opt_type='MVol', optimizer=solver, num_trys=20,
                                                min_weight_constraint=min_weight_constraint)
                if weights_MVol[0] == 0:
                    break
            return weights_MVol[2]

        elif output in ['InvVar', 'IVar']: # Inverse Variance
            tmp_VarS = pd.Series(np.diag(CovS), index=[CovS.index, CovS.columns])
            weights_INV = dict(zip(list(returns_DF.columns), inv_weigts(tmp_VarS)))
            return weights_INV

        elif output in ['InvVol', 'IVol', 'InvVola', 'IVola', 'IStd', 'InvStd']: # Inverse Volaitility
            tmp_StdS = pd.Series(np.diag(CovS), index=[CovS.index, CovS.columns]).pow(1/2)
            weights_INV = dict(zip(list(returns_DF.columns), inv_weigts(tmp_StdS)))
            return weights_INV

    elif output in ['1/n', '1/N', 'equal', 'naive']:
        equal_weights = np.ones(len(returns_DF.columns.tolist()))
        equal_weights = equal_weights / sum(equal_weights)
        return dict(zip(returns_DF.columns.tolist(), equal_weights))


    if min_weight_constraint and output not in ['MSR', 'EF', 'MVol', 'MinV', 'MinVola']:
        raise Exception('Min_weigh_constraint for this output type not implemented!')


def PF_value_with_rebalancing(stock_return_df, stock_weights,initial_investment=1, rebalancing='daily', percentage=False):
    """
    This function calculates the Portfolio value/return given stock returns and stock weights considering rebalancing.

    Args:
        stock_return_df (pd.DataFrame): returns of single stocks
        stock_weights (dict/np.Array): array or dictionary of weights
        initial_investment (int/float): initial investment volume
        rebalancing ('d'/'m'/'y'): rebalancing frequency (options: 'daily', 'monthly', 'yearly')
        percentage (bool): returns percentage returns instead of dollar amounts

    Returns:
        DataFrame of portfolio positions with total portfolio value column 'PF Total'

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """

    if type(stock_weights)==list:
        stock_weights = np.asarray(stock_weights)
    elif type(stock_weights)==dict:
        stock_return_df = stock_return_df[stock_weights.keys()]
        stock_weights = np.asarray(list(stock_weights.values()))

    stock_return_array = stock_return_df.to_numpy() + 1

    current_PF_value = initial_investment
    PF_performance = []
    last_date = pd.to_datetime(stock_return_df.index[0]) - datetime.timedelta(days=-(1+365+31))
    for idx, row in zip(pd.to_datetime(stock_return_df.index), stock_return_array):
        if rebalancing == 'daily' or rebalancing == 'd':
            tmp = (row * stock_weights) * current_PF_value
        elif rebalancing == 'monthly' or rebalancing == 'm':
            if idx.month != last_date.month:
                tmp = (row * stock_weights) * current_PF_value
            else:
                tmp = row * last_row
        elif rebalancing == 'yearly' or rebalancing == 'y':
            if idx.year != last_date.year:
                tmp = (row * stock_weights) * current_PF_value
            else:
                tmp = row * last_row
        else:
            raise Exception('Not defined rebalancing.')
        last_row = tmp
        last_date = idx
        current_PF_value = tmp.sum()
        tmp = np.append(tmp, current_PF_value)
        PF_performance.append(tmp.tolist())
    PF_performance = pd.DataFrame(PF_performance, columns=(stock_return_df.columns.tolist() + ['PF Total']))
    PF_performance['Date'] = stock_return_df.index.tolist()
    PF_performance.set_index('Date', inplace=True)
    if percentage == True:
        tmp = PF_performance.iloc[0] / (np.append(stock_weights, 1) * initial_investment)
        PF_performance = PF_performance.pct_change()
        PF_performance.iloc[0] = tmp - 1

    return PF_performance


def moments(returns, output='all', r_f=0, days_pa=252, Cov_matrix=False):
    """
    Calculate moments like Mean, Variance, etc. of return data.

    Args:
        returns (pd.DataFrame): daily returns DataFrame
        output (str/list): 'all' returns all moments, a list of moments e.g. ['Mean (p.a.)', 'Var (p.a.)'] just returns the respective moments
        r_f (float): risk-free rate (needed for Sharpe Ratio)
        days_pa (int): number of days per anno
        Cov_matrix (bool): if True also returns covariance matrix

    Returns:
        DataFrame with each column's moments

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    # All metrics are arithmetically not geometrically.
    stats=pd.DataFrame()
    stats["Mean"]=returns.mean()
    stats["Mean (p.a.)"]=stats["Mean"] * days_pa
    stats["Std"]=returns.std(ddof=0)
    stats["Std (p.a.)"]=returns.std(ddof=0) * np.sqrt(days_pa)
    stats["SStd"]=returns.std(ddof=1)
    stats["SStd (p.a.)"]=returns.std(ddof=1) * np.sqrt(days_pa)
    stats["Var"]=returns.var(ddof=0)
    stats["Var (p.a.)"]=returns.var(ddof=0) * days_pa
    stats["SVar"]=returns.var(ddof=1)
    stats["SVar (p.a.)"]=returns.std(ddof=1) * days_pa
    stats["Skew"]=returns.skew()
    stats["Kurt"]=returns.kurtosis()
    def MDD(returns):
        prices = (returns + 1).cumprod()
        previous_peaks = prices.cummax()
        drawdown = prices / previous_peaks - 1
        MDD = drawdown.min()
        return MDD
    stats["MaxD"]=MDD(returns) #Maximum drawdown
    stats["Sharpe (p.a.)"]=(stats["Mean (p.a.)"]-r_f)/stats["SStd (p.a.)"] # Sharpe Ratio
    stats["Sharpe"]=(stats["Mean"]-(r_f/days_pa))/stats["SStd"] # Sharpe Ratio
    stats["CE"] = np.exp(np.log(1 + returns).mean(axis=0)) - 1 #Certainty Equivalent
    if output != 'all':
        if type(output) != list:
            output = list(output)
        if isinstance(stats, pd.DataFrame):
            stats = stats[output]
        elif isinstance(stats, pd.Series):
            stats = stats.loc[output]
        else:
            raise Exception('Unknown data type.')

    if Cov_matrix == True:
        if version.parse(pd.__version__) >= version.parse("1.1"):
            return stats, returns.cov(ddof=1), returns.cov(ddof=1) * days_pa
        else:
            return stats, returns.cov(), returns.cov() * days_pa
    else:
        return stats


class df_date_period_iterator:
    """
    Iterate across DataFrame with date as index.

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    def __init__(self, df, period='q', previous_periods=False, true_date=False):
        """
        Iterate across DataFrame with date as index.

        Args:
            df (pd.DataFrame): DataFrame with date-index to iterate across
            period ('d','m','q','y'): iterating date frequency
            previous_periods (bool): False = rolling (keep only one period) / True = adding new date (append new quarter t previous)
            true_date (bool): False - return e.g. 'yyyymm' key / True - return real datetime

            .. _Google Python Style Guide:
                http://google.github.io/styleguide/pyguide.html

        """
        self.previous_periods = previous_periods
        self.period = period
        self.true_date = true_date
        df = df.copy()
        df['year'] = df.index.strftime("%Y").astype(int)
        df['month'] = df.index.strftime("%m").astype(int)
        df['quater'] = (df['month']/3).apply(np.ceil).astype(int)
        df['day'] = df.index.strftime("%d").astype(int)
        if period == 'q':
            df['iter'] = (df['year'].astype(str) + df['quater'].astype(str).apply(lambda x: x.zfill(2))).astype(int)
        elif period == 'm':
            df['iter'] = (df['year'].astype(str) + df['month'].astype(str).apply(lambda x: x.zfill(2))).astype(int)
        elif period == 'd':
            df['iter'] = (df['year'].astype(str) + df['month'].astype(str).apply(lambda x: x.zfill(2)) + df['day'].astype(str).apply(lambda x: x.zfill(2))).astype(int)
        elif period == 'y':
            df['iter'] = df['year'].astype(int)
        else:
            raise Exception("Not defined period method.")

        self.df = df
        self.date_list = list(df['iter'].unique())
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num == len(self.date_list):
            raise StopIteration
        if self.previous_periods == False: # Just one period
            return_df = self.df.loc[self.df['iter'] == self.date_list[self.num]]
        elif self.previous_periods == True: # Current period and all previous
            return_df = self.df.loc[self.df['iter'] <= self.date_list[self.num]]
        self.num += 1
        # Output real datetime as dt_out
        dt = self.date_list[self.num - 1]
        if self.true_date == True:
            if self.period == 'q':
                year = int(str(dt)[:4])
                day = 1
                if int(str(dt)[4:]) * 3 == 12:
                    year += 1
                    month = 1
                else:
                    month = int(str(dt)[4:]) * 3 + 1
                dt_out = datetime.date(year, month, day) - datetime.timedelta(days=1)
            elif self.period == 'm':
                year = int(str(dt)[:4])
                day = 1
                if int(str(dt)[4:]) == 12:
                    year += 1
                    month = 1
                else:
                    month = int(str(dt)[4:]) + 1
                dt_out = datetime.date(year, month, day) - datetime.timedelta(days=1)
            elif self.period == 'd':
                dt = str(dt)
                dt_out = datetime.date(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            elif self.period == 'y':
                dt_out = datetime.date(dt+1, 1, 1) - datetime.timedelta(days=1)
        else:
            dt_out = dt
        return dt_out, return_df[return_df.columns.difference(['year', 'month', 'quater', 'day', 'iter'])]


def transpose_dict_DF(DF_dict, change_list_touples):
    """
    Transposes a 3D data object of an dict of DataFrames.

    Args:
        DF_dict (dict of pd.Dataframe): 3D data object which is an dictionary of DataFrames
        change_list_touples (list of tuples): transpose list of swap touples e.g. [('head','index'),('dict_key','index')] meaning first swap DF columns DF rows and then swap the dictionary keys with the DF rows

    Returns:
        Transposed dictionary of DataFrames

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """

    temp_dict_DF = DF_dict
    for swap_old, swap_new in change_list_touples:

        def swap_heads_dictkey(temp_dict_DF, transpose=False):
            if transpose:
                new_dictDF = {}
                for dict_key, DF in temp_dict_DF.items():
                    new_dictDF[dict_key] = DF.transpose()
                temp_dict_DF = new_dictDF
            # Get all keys of all 3 dimensions
            dict_keys = []
            heads = []
            indices = []

            for dict_key, DF in temp_dict_DF.items():
                dict_keys.append(dict_key)
                heads.extend(list(DF.columns.values))
                indices.extend(list(DF.index.values))
            dict_keys = list(dict.fromkeys(dict_keys))
            heads = list(dict.fromkeys(heads))
            indices = list(dict.fromkeys(indices))

            # Sort indices if sort-able
            if isinstance(indices[0], datetime.date) or type(indices[0]) == int:
                indices = indices.sort()
            if isinstance(heads[0], datetime.date) or type(heads[0]) == int:
                heads = heads.sort()
            if isinstance(dict_keys[0], datetime.date) or type(dict_keys[0]) == int:
                dict_keys = dict_keys.sort()

            # SWAP heads/columns with dict_keys
            new_dictDF = {}
            for dict_key in heads:
                temp_DF = pd.DataFrame(columns=dict_keys, index=indices)
                for key, DF in temp_dict_DF.items():
                    temp_DF[key] = DF[dict_key].rename(key)
                new_dictDF[dict_key] = temp_DF

            temp_dict_DF = new_dictDF

            if transpose:
                new_dictDF = {}
                for dict_key, DF in temp_dict_DF.items():
                    new_dictDF[dict_key] = DF.transpose()
                temp_dict_DF = new_dictDF

            return temp_dict_DF


        if (swap_old == 'head' and swap_new == 'index') or (swap_old == 'index' and swap_new == 'head'):
            new_dictDF = {}
            for dict_key, DF in temp_dict_DF.items():
                new_dictDF[dict_key] = DF.transpose()

        elif (swap_new == 'dict_key' and swap_old == 'index') or (swap_new == 'index' and swap_old == 'dict_key'):
            new_dictDF = swap_heads_dictkey(temp_dict_DF, transpose=True)


        elif (swap_new == 'dict_key' and swap_old == 'head') or (swap_new == 'head' and swap_old == 'dict_key'):
            new_dictDF = swap_heads_dictkey(temp_dict_DF, transpose=False)


        temp_dict_DF = new_dictDF

    return temp_dict_DF


def print_nice_data_tables(data, style='simple', heading=True, max_rows=50):
    """
    This function prints a DataFrame/Dictionary/etc. and a dictionary of DFs nicely.

    Args:
        data (pd.Dataframe/dict of pd.DataFrame): data to print (DataFrame or dictionary of DFs)
        style (str): table style from tabulate
        heading (bool): show 'Date Tables:' head

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    def shorten_DF(DF, max_rows):
        tmp_DF = DF.copy()
        if max_rows != False:
            if len(DF) > max_rows:
                tmp_DF = tmp_DF.iloc[:int(max_rows/2)+1]
                #for col in list(DF.columns):
                #    tmp_DF.iloc[len(tmp_DF)-1][col] = "–––––"
                tmp_DF = tmp_DF.append(DF.iloc[-int(max_rows/2):])
                print('* Table output was shrtened to {} rows.\n'.format(str(max_rows)))
        return tmp_DF
        
    if type(data)==dict:
        if heading:
            print('-'*30)
            print('Data Tables:')
            print('-' * 30)
        for key, DF in data.items():
            print('\n' + key + ':')
            print(tabulate(shorten_DF(DF, max_rows), tablefmt=style, headers="keys", numalign="right", floatfmt=".4f") + '\n')
                
    else:
        if heading:
            print('-' * 30)
            print('Data Table:')
            print('-' * 30)
        print(tabulate(shorten_DF(data, max_rows), tablefmt=style, headers="keys", numalign="right", floatfmt=".4f") + '\n')


def custom_plot(data, style='2-col', mean_row_loc=False, print_data='last', title="Please specify a 'title'.", xlabel=None, ylabel=None):

    if print_data=='first':
        print_nice_data_tables(data, style='simple')


    length = len(data)
    my_colors = ["#999999", "#FFBE0F", "#3E91CC", "#949494", "#3498db", "#ffc93c", "#ff4301", "#28abb9", "#52575d", "#d54062"]# erste 3 Farben neu eingefügt
    # other colors ['#3E91CC', "#949494", "#FFBE0F",'#F0A400', "#CDCDCD", "#FFBE0F"]
    my_cmap = matplotlib.colors.ListedColormap(sns.color_palette(my_colors).as_hex())
    try:
        plt.close()
    finally:
        if style=='one':


            data.plot(kind='line', legend=True, title=title, cmap=my_cmap, xlabel=xlabel, ylabel=ylabel)
            

            plt.show()

        elif style=='2-col':
            fig, axs = plt.subplots(math.ceil(length/2), 2, figsize=(10,(5*math.ceil(length/2))))
            i = 0
            j = 0
            for name, DF in data.items():
                if mean_row_loc != False:
                    mean_row = DF.loc[mean_row_loc]
                    DF = DF.drop(index=mean_row_loc)
                DF.plot(kind='line', ax=axs[i, j], xlim=[DF.index.min(),DF.index.max()], legend=False, title=name, cmap=my_cmap)
                plt.setp(axs[i, j].xaxis.get_majorticklabels(), rotation=45)
                if mean_row_loc != False:
                    m = 0
                    for col in list(mean_row.index):
                        axs[i, j].axhline(y=float(mean_row[col]), color=axs[i, j].get_lines()[m].get_color(), linestyle='dashed', linewidth=1)

                        m += 1
                j += 1
                if j > 1:
                    i +=1
                    j = 0
            if math.ceil(length/2) != length/2:
                fig.delaxes(axs[math.ceil(length/2)-1, 1])

            plt.tight_layout()  # This to avoid overlap of labels and titles

            plt.subplots_adjust(bottom=0.18)
            lines, labels = fig.axes[0].get_legend_handles_labels()
            if mean_row_loc != False:
                labels.extend([mean_row_loc])
                mean_line = plt.plot(1, color='black', linestyle='dashed')
                lines.extend(mean_line)
            fig.legend(lines, labels, loc='lower center')

            plt.show()

        if print_data == 'last':
            print_nice_data_tables(data, style='simple')


def excel_exporter(data, file='export.xlsx'):
    """
    Exports a DataFrame or a dictionary of DataFrames to an excel (.xlsx) file.

    Args:
        data (pd.DataFrame/dict of DataFrames): data to export whcih can be a DataFrame or a dictionary of DataFrames
        file (str): file path and name of excel export file

    .. _Google Python Style Guide:
        http://google.github.io/styleguide/pyguide.html

    """
    writer = pd.ExcelWriter(file)
    if type(data) == dict:
        for name, df in data.items():
            # Remove invaild characters from sheet name
            name = name.translate({ord(name): " " for name in "!@#$%^&*()[]{};:,/<>?\|`~="})
            # Max Excel sheet name length 31 characters
            name = (name[:29] + '..') if len(name) > 29 else name
            df.to_excel(writer, sheet_name=name)
        writer.save()
    else:
        data.to_excel(writer, sheet_name='sheet1')
        writer.save()




def VaR(returns_DF, p=0.95, method='hist', output='VaR', n_days=1, n_sims = 1000000, random_seed=42, ewma=False, period_days=252):
    #if method == 'hist':
    returns_DF = returns_DF.sort_index()
    def historical(returns_DF, p=p, n_days=n_days, ewma=ewma, period_days=period_days):

        my_list = list(returns_DF.columns)
        if ewma == False:
            returns_DF['prob'] = 1/len(returns_DF)
        else:
            # With weekends
            # returns_DF['days'] = (returns_DF.index.max() - returns_DF.index).days 
            # Without weekends
            returns_DF['days'] = np.arange(len(returns_DF)-1, -1,-1)
            # Note: np.log is ln
            lamda = 1 - (np.log(2)/(ewma*period_days))
            # print('lambda:', lamda)
            # Formula used: (1-λ)× λ^(t-1)
            returns_DF['prob'] = (1-lamda) * lamda ** (returns_DF['days'])
            
            # Just to be sure that sum of prob is 1 but just needed if with weekends
            returns_DF['prob'] = returns_DF['prob'] / returns_DF['prob'].sum()    

        # Order DF by size
        VaR = {}
        ES = {}


        for col in my_list:
            tmp_DF = returns_DF[[col,'prob']].sort_values(by=col)
            tmp_DF.index = np.arange(1, len(tmp_DF) + 1)
            tmp_DF['c_prob'] = tmp_DF['prob'].cumsum()
        
            for idx, row in tmp_DF.iterrows():
                if row['c_prob']  >= (1-p):
                    last_row = idx 
                    break

            first_row = last_row - 1
            last_figure = tmp_DF.iloc[last_row][col]

            #print('Frist row:',first_row,'Last row:',last_row)

            if first_row == 0:
                tmp_VaR = last_figure
                tmp_ES = last_figure
            else:
                first_figure = tmp_DF.iloc[first_row][col]
            
                q = ((1-p) - tmp_DF.loc[first_row]['c_prob'])/(tmp_DF.loc[last_row]['c_prob'] - tmp_DF.loc[first_row]['c_prob'])
                tmp_VaR = (first_figure * q + last_figure * (1-q)) * np.sqrt(n_days)
            
                tmp_DF = tmp_DF.loc[:last_row]
                tmp_DF.at[last_row, 'prob'] =  tmp_DF.at[last_row, 'prob'] - (tmp_DF['prob'].sum() - (1-p))

                tmp_ES=(tmp_DF['prob'] * np.sqrt(n_days) * tmp_DF[col]).sum()
                
            
            VaR[col] = tmp_VaR
            ES[col] = tmp_ES
            
        VaR = pd.Series(VaR).rename('VaR')
        ES = pd.Series(ES).rename('ES')
       
        VaR = VaR * (-1)
        ES = ES * (-1)

        return VaR, ES

    #elif method in ['simulation', 'MC', 'Monte Carlo', 'Simulation']:
    def simulation(returns_DF, n_days=n_days, p=p, n_sims=n_sims, random_seed=random_seed):
        mean = returns_DF.mean() * n_days
        std = returns_DF.std(ddof=1) * np.sqrt(n_days)
        VaR = pd.Series().rename('VaR')
        ES = pd.Series().rename('ES')
        for col in returns_DF.columns.to_list():
            np.random.seed(random_seed)
            sim_returns = np.random.normal(mean[col], std[col], n_sims)
            VaR[col] = np.percentile(sim_returns, (1-p)*100)
            ES[col] = sim_returns[sim_returns < VaR[col]].sum() / (p * n_sims)
        VaR = VaR * (-1)
        ES = ES * (-1)
        return VaR, ES


    #elif method in ['delta-normal', 'parametric', 'normal']:
    def delta_normal(returns_DF, n_days=n_days, p=p):
        mean = np.mean(returns_DF) * n_days
        std = np.std(returns_DF, ddof=1) * np.sqrt(n_days)
        Z_99 = norm.ppf(1 - p)
        VaR = (-mean - (std * Z_99)).rename('VaR')
        ES = ((1-p) ** (-1) * norm.pdf(norm.ppf(1-p)) * std - mean).rename('ES')
        return VaR, ES


    if type(output) == list and type(method) == list:
        raise Exception('Either output or method can be a list not both.')
    elif type(output) == list:
        if method in ['delta-normal', 'parametric', 'normal']:
            VaR, ES = delta_normal(returns_DF)
        elif method in ['simulation', 'MC', 'Monte Carlo', 'Simulation']:
            VaR, ES = simulation(returns_DF)
        elif method in ['hist']:
            VaR, ES = historical(returns_DF)
        else:
            raise Exception('Not defined moethod.')
        return pd.concat([VaR, ES], axis=1)
    elif type(method) == list:
        exp_list = []
        if output == 'VaR':
            for my_method in method:
                if my_method in ['delta-normal', 'parametric', 'normal']:
                    VaR, _ = delta_normal(returns_DF)
                elif my_method in ['simulation', 'MC', 'Monte Carlo', 'Simulation']:
                    VaR, _ = simulation(returns_DF)
                elif my_method in ['hist']:
                    VaR, _ = historical(returns_DF)
                else:
                    raise Exception('Not defined moethod.')
                exp_list.append(VaR.rename(my_method))
        elif output == 'ES':
            for my_method in method:
                if my_method in ['delta-normal', 'parametric', 'normal']:
                    _, ES = delta_normal(returns_DF)
                elif my_method in ['simulation', 'MC', 'Monte Carlo', 'Simulation']:
                    _, ES = simulation(returns_DF)
                elif my_method in ['hist']:
                    _, ES = historical(returns_DF)
                else:
                    raise Exception('Not defined moethod.')
                exp_list.append(ES.rename(my_method))
        else:
            raise Exception('Not defined output.')
        return pd.concat(exp_list, axis=1)



def drop_suffix(df, suffix):
    df.columns = df.columns.str.rstrip(suffix)
    return df

def check_VaR_violations(returns_df, VaR_df, NA_method='ffill'):
    df_returns = returns_df.copy()
    df_returns.index = pd.to_datetime(df_returns.index)
    df_VaR = VaR_df.copy()
    df_VaR.index = pd.to_datetime(df_VaR.index)
    original_cols = list(df_returns.columns)
    if sorted(list(df_returns.columns)) != sorted(list(df_VaR.columns)):
        print('Actual DF:', sorted(list(df_returns.columns)))
        print('Limit DF:', sorted(list(df_VaR.columns)))
        raise(Exception('Both data frames have not the same columns!'))
    df_returns.columns = [str(col) + '_actual' for col in df_returns.columns]
    df_VaR.columns = [str(col) + '_limit' for col in df_VaR.columns]
    actual_columns = list(df_returns.columns)
    limit_columns = list(df_VaR.columns)
    combined = pd.concat([df_returns, df_VaR], axis=1)
    combined[limit_columns] = combined[limit_columns].fillna(method=NA_method)
    combined = combined.dropna()
    breaches = drop_suffix(combined[actual_columns], '_actual')[original_cols] > drop_suffix(combined[limit_columns], '_limit')[original_cols]
    #print(VaR_df.index)

    out_breach = pd.DataFrame(columns=["Date","Portfolio","Actual Loss","Est. VaR"])
    for breach in breaches.iterrows():
        idx, breach = breach
        if breach.any():
            for PF, value in breach.iteritems():
                if value:
                    out_breach = out_breach.append({"Date":breach.name,"Portfolio":PF,"Actual Loss":combined.at[idx, PF+'_actual'],"Est. VaR":combined.at[idx, PF+'_limit']}, ignore_index=True)
    concl_breaches = pd.DataFrame(columns=['PF','# VaR Exceeded',"Actual avg. loss","Average est. VaR",'Percentage','Observations'])
    for col in original_cols:
        count = out_breach[out_breach['Portfolio'] == col]['Portfolio'].count()
        breach = out_breach[out_breach['Portfolio'] == col]["Actual Loss"].mean()
        limit = out_breach[out_breach['Portfolio'] == col]["Est. VaR"].mean()
        length = breaches.shape[0]
        pct = count / length
        concl_breaches = concl_breaches.append(pd.Series({'PF':col, '# VaR Exceeded':count,"Actual avg. loss":breach,"Average est. VaR":limit,'Percentage':pct,'Observations':length}), ignore_index=True)
    concl_breaches.set_index('PF', inplace=True)
    return out_breach, concl_breaches

