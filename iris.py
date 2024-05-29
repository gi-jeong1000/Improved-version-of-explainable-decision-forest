from ExperimentSetting import *
import warnings

warnings.filterwarnings('ignore')
branch_probability_thresholds=[10]
filter_approaches = ['probability']
df_names = ['iris']
number_of_estimators=100
fixed_params={}
num_of_iterations=1


e = ExperimentSetting(branch_probability_thresholds,df_names,
                                     number_of_estimators,fixed_params,num_of_iterations)
e.run()


