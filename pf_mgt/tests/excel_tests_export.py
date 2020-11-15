import sys, os
import datetime as dt

sys.path.append(os.path.dirname(os.path.abspath('../..')))
from assignment.our_modules import *


def sample_return_DF():
    stock_a = pd.Series({dt.date(2020,1,1):0.02, dt.date(2020,1,2):0.05, dt.date(2020,1,3):0.01}).rename('Stock A')
    stock_b = pd.Series({dt.date(2020, 1, 1): 0.07, dt.date(2020, 1, 2): 0.13, dt.date(2020, 1, 3): 0.01}).rename('Stock B')
    stock_c = pd.Series({dt.date(2020, 1, 1): 0.01, dt.date(2020, 1, 2): 0.10, dt.date(2020, 1, 3): 0.02}).rename('Stock B')
    merged = pd.concat([stock_a, stock_b, stock_c], axis=1)
    return merged


def pf_optimal_weights(returns):
    exp_dict = {}
    MSR = pd.Series(PF_weights(returns, output='MSR')).rename('MSR')
    MVol = pd.Series(PF_weights(returns, output='MVol')).rename('Min Vola')
    naive = pd.Series(PF_weights(returns, output='1/N')).rename('1/N')
    IVola = pd.Series(PF_weights(returns, output='IVola')).rename('Inverse Vola')
    IVar = pd.Series(PF_weights(returns, output='IVar')).rename('Inverse Variance')
    exp_dict['weights'] = pd.concat([MSR, MVol, naive, IVola, IVar])
    exp_dict['stocks'] = returns
    return exp_dict



def import_DF_for_test_cases():
    fname = os.path.abspath('../..') + "/portfolio management assignment/data/Assignments2020_QP&PRM_05-10-2020.xlsm"

    df1 = get_data_nice_looking(fname, 'ETF & FF Factor Time Series', sheet_props={'index_col': 0, 'header': 0, 'parse_dates':'index'})
    return df1

def exp_PF_value_with_rebalancing(df, output_path):
    benchmark_weight_dict = {'Wilshire 5000 Total Market Index (^W5000) $': 0.6, '13 Week Treasury Bill (^IRX) $': 0.4}
    exp_dict = {'INPUT weights':pd.DataFrame.from_dict(benchmark_weight_dict, orient='index'), 'INPUT data': df}
    exp_dict['OUT daily rebalancing'] = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000, rebalancing='daily', percentage=False)
    exp_dict['OUT daily rebal percent'] = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000, rebalancing='daily', percentage=True)
    #exp_dict['OUT qrt rebalancing'] = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000,
    #                                                              rebalancing='quaterly', percentage=False)
    exp_dict['OUT monthl rebalancing'] = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000,
                                                                  rebalancing='monthly', percentage=False)
    exp_dict['OUT yearly rebal percent'] = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000,
                                                                  rebalancing='yearly', percentage=True)
    excel_exporter(exp_dict, output_path + '/PF value w rebalancing.xlsx')
    return "PF value rebalancing exported"
    #output_path = output_path + '/PF value w rebalancing'
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)


def exp_moments(df, output_path):
    benchmark_weight_dict = {'Wilshire 5000 Total Market Index (^W5000) $': 0.6, '13 Week Treasury Bill (^IRX) $': 0.4}
    exp_dict = {'INPUT weights':pd.DataFrame.from_dict(benchmark_weight_dict, orient='index'), 'INPUT data': df}
    mom = moments(PF, r_f=0.02, days_pa=252)
    exp_dict['OUT 252d r0_02'] = mom
    mom = moments(PF, r_f=0, days_pa=100)
    exp_dict['OUT 100d r0_00'] = mom
    excel_exporter(exp_dict, output_path + '/PF moments.xlsx')
    return "PF value rebalancing exported"

def exp_iteraor_date(df, freq, output_path):
    output_path = output_path + '/data seperator by date'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    exp_dict = {'INPUT_'+str(freq):df}
    for date, data in df_date_period_iterator(df, period=freq, previous_periods=False, true_date=False):
        exp_dict[str(date)] = data
    excel_exporter(exp_dict, output_path + '/Date iterator_'+str(freq)+'_rolling.xlsx')
    exp_dict = {'INPUT_' + str(freq): df}
    for date, data in df_date_period_iterator(df, period=freq, previous_periods=True, true_date=True):
        exp_dict[str(date)] = data
    excel_exporter(exp_dict, output_path + '/Date iterator_' + str(freq) + '_adding.xlsx')


def exp_transpse_3D(df, transposing, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df = df[['Invesco Global Listed Private Equity ETF (PSP) $','Small-minus-Big (SMB)', '13 Week Treasury Bill (^IRX) $']]
    imput_dict = {}
    for date, data in df_date_period_iterator(df, period='d', previous_periods=False, true_date=False):
        imput_dict[str(date)] = moments(data)
    for date, data in df_date_period_iterator(df, period='m', previous_periods=False, true_date=False):
        imput_dict[str(date)] = moments(data)
    for date, data in df_date_period_iterator(df, period='q', previous_periods=False, true_date=False):
        imput_dict[str(date)] = moments(data)
    imput_dict = {key: imput_dict[key] for key in ['20200330', '20200331', '20200401', '202004', '202003', '202001', '202002']}
    export_dict = transpose_dict_DF(imput_dict, transposing)
    excel_exporter(imput_dict, output_path + '/INPUT 3D data.xlsx')
    excel_exporter(export_dict, output_path + '/OUTPUT 3D data.xlsx')


def exp_currency_conversion(DF, output_path):
    output_path = output_path + '/currency conversion.xlsx'
    export_dict= {}
    df1 = DF_column_currency_conversion(DF.loc['2013-07-01':'2020-08-17'], ['iShares STOXX Europe 600 UCITS ETF (DE) (EXSA.DE) €'],
                                        DF['USD/EUR (EUR=X) €'].loc['2020-07-01':])
    df2 = DF_column_currency_conversion(DF.loc['2020-07-01':], ['iShares Core Nikkei 225 ETF (1329.T) ¥'], DF['USD/JPY (JPY=X) ¥'].loc['2013-07-01':'2020-08-17'])
    export_dict['convert iShares STOXX Europe'] = df1
    export_dict['convert Nikkei'] = df2
    export_dict['INPUT']= DF
    excel_exporter(export_dict, output_path)



if __name__ == '__main__':
    df = import_DF_for_test_cases().pct_change()
    df = df.drop(df.index[0])
    df = df.drop(df.index[-1])
    df.fillna(0, inplace=True)
    #print(df[['Wilshire 5000 Total Market Index (^W5000) $', '13 Week Treasury Bill (^IRX) $']])
    test_output = os.path.abspath('..') + '/tests/excel'
    if not os.path.exists(test_output):
        os.makedirs(test_output)
    exp_PF_value_with_rebalancing(df, test_output)

    excel_exporter(pf_optimal_weights(sample_return_DF()), test_output + '/pf_weights.xlsx')

    benchmark_weight_dict = {'Wilshire 5000 Total Market Index (^W5000) $': 0.6, '13 Week Treasury Bill (^IRX) $': 0.4}
    PF = PF_value_with_rebalancing(df, benchmark_weight_dict, initial_investment=100000, rebalancing='daily', percentage=True)
    exp_moments(PF, test_output)

    exp_iteraor_date(df.loc['2018-06-01':], 'q', test_output)
    exp_iteraor_date(df.loc['2018-06-01':], 'm', test_output)
    exp_iteraor_date(df.loc['2020-06-01':], 'd', test_output)

    exp_currency_conversion(df, test_output)

    test_output = test_output + '/transpose 3D data strcuture'
    exp_transpse_3D(df.loc['2020-01-01':], transposing=[('head', 'dict_key')], output_path =test_output + '/swap_columns_with_sheets')
    exp_transpse_3D(df.loc['2020-01-01':], transposing=[('index', 'dict_key')],
                    output_path=test_output + '/swap_rows_with_sheets')
    exp_transpse_3D(df.loc['2020-01-01':], transposing=[('head', 'index')], output_path =test_output + '/swap_columns_with_rows')
    exp_transpse_3D(df.loc['2020-01-01':], transposing=[('head', 'dict_key'), ('index', 'dict_key')], output_path =test_output + '/swap_columns_with_sheets_and then_rows_with_sheets')



