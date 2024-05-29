import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import glob
import pickle

f_paths = glob.glob('pickles_100trees/*')
results=[]
for path in tqdm(f_paths):
    results.append(pickle.load(open(path,'rb')))

df = pd.DataFrame(results)
groups = ['df_name']
df[[col for col in df.columns if 'old' in col]+groups].groupby(groups).mean()

df = pd.DataFrame(results)

#groups = ['df_name','minimum_estimators','exclusion_threshold','max_number_of_branches']
groups = ['df_name','max_number_of_branches']
df[[col for col in df.columns if 'average_depth' in col]+groups].groupby(groups).mean()

print(df)