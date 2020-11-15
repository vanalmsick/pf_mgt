import pytest
import datetime as dt
import pandas as pd
import numpy as np
import os, sys, pathlib, inspect

# Import all portfolio management assignment-specific functions that our group implemented
sys.path.append(os.path.dirname(os.path.abspath('../..')))
from assignment.our_modules import *


def sample_return_DF():
    stock_a = pd.Series({dt.date(2020,1,1):0.02, dt.date(2020,1,2):0.05, dt.date(2020,1,3):0.01}).rename('Stock A')
    stock_b = pd.Series({dt.date(2020, 1, 1): 0.07, dt.date(2020, 1, 2): 0.13, dt.date(2020, 1, 3): 0.01}).rename('Stock B')
    stock_c = pd.Series({dt.date(2020, 1, 1): 0.01, dt.date(2020, 1, 2): 0.10, dt.date(2020, 1, 3): 0.02}).rename('Stock B')
    merged = pd.concat([stock_a, stock_b, stock_c], axis=1)
    return merged



def test_pf_weights():
    DF = sample_return_DF()
    weights_MSR = PF_weights(DF, output='MSR')
    weights_MSR = dict(zip(weights_MSR.keys(), np.round(weights_MSR.keys(), decimals=2)))
    assert weights_MSR == {'Stock A': 0.98, 'Stock B': 0.01, 'Stock C': 0.01}


# Export our sample returns DF to .CSV
#sample_return_DF().to_csv('test_retuns.csv', sep=';', decimal=',')
