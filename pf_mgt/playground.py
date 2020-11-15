import pandas as pd
from our_modules import *


df = pd.read_csv('etp.csv',sep=';',decimal=',',index_col=0)
print(df)

print(VaR(df, p=0.95, method=['hist','MC','normal'], output='VaR', n_days=1, n_sims=1000000, random_seed=42))
#print(VaR(df, p=0.95, method='MC', output=['VaR', 'ES'], n_days=1, n_sims=1000000, random_seed=42))
#print(VaR(df, p=0.95, method='normal', output=['VaR', 'ES'], n_days=1, n_sims=1000000, random_seed=42))