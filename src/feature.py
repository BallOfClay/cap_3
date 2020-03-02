import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import csv
import sys, os
sys.path.append('~/dsi/capstones/cap_3/')

# import scipy as stats
import re
import dateparser
import datetime
import math
import json

# from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DateType
# import pyspark as ps

# import researchpy as rp
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

from common import is_bool_dtype
from clean import parse_date
from clean import refactor_time

# from operator import attrgetter
# df['duration_dataset'] = (
#     df['date_1'].dt.to_period('M') -
#     df['date_2'].dt.to_period('M')).apply(attrgetter('n'))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

class FEATURE(object):
    '''
    For a given feature, and given device type, class determines:
        which device has this feature
        when each oem released the feature
        when oem released feature in respects to first feature release
        when each oem removed the feature 
        when oem removed feature in respects to first feature removal
    '''

    def __init__(self, df, feature, device):
        '''
        Parameters
        ----------
        df : data-frame
            The data-frame to parse features from.
        feature : string-like
            Name of feature to be parsed
        device : string-like
            string denoting 'phone', 'tablet', 'watch'
        ''' 

        self.df = df
        self.feature = feature
        self.device = device
        self.col_list = ['oem', # 'oem_first_release',
                        'feat_released_month', 'months_after_release', 
                        'feat_removed_month', 'months_after_removal']

        # self.df_device = None
        self.df_part = pd.DataFrame()
        self.df_feat = self._company_release()
        
    def _clean_df(self):
        '''
        Clean dataframe prior to parsing
        '''

        if 'Unnamed: 0' in self.df.columns:
           self.df.drop(columns=['Unnamed: 0'], inplace=True) 


    def _choose_device(self):
        '''
        Choose device type to encode
        '''

        if self.device == 'tablet':
            self.df = self.df.loc[self.df['is_tablet'], :]
        elif self.device == 'watch':
            self.df = self.df.loc[self.df['is_watch'], :]
        else:
            df_step = self.df.loc[~self.df.is_tablet, :]
            df_step = df_step.loc[~df_step.is_watch, :]
            self.df = df_step[df_step['network_technology'].notna()]
        # return self.df
    

    def _init_df_feat(self):
        '''
        Initializes Feature specific Data-Frame
        '''

        man_list = list(self.df['oem'].unique())
        self.df_feat = pd.DataFrame(data=np.full([len(man_list), len(self.col_list)], 
                                                fill_value = np.nan),
                                            columns = self.col_list)
        self.df_feat['oem'] = man_list


    def _company_release(self):
        '''
        Determines which companies have released and removed a given feature, and on which months.
        '''

        self._clean_df()
        self._choose_device()
        self._init_df_feat()

        man_list = list(self.df['oem'].unique())
        
        for man in man_list:
            df_step = self.df[self.df['oem'] == str(man)].sort_values(by='launch_announced')
            
            if is_bool_dtype(df_step[str(self.feature)]):
                first_idx = df_step[df_step[str(self.feature)]==True].first_valid_index()
                last_idx = df_step[df_step[str(self.feature)]==True].last_valid_index()
                if first_idx == None:
                    pass
                else:
                    self.df_feat.loc[self.df_feat['oem'] == str(man), ['feat_released_month']] = df_step.loc[first_idx, 'launch_announced']

                    self.df_feat.loc[self.df_feat['oem'] == str(man), ['feat_removed_month']] = df_step.loc[last_idx, 'launch_announced']

            else:
                first_idx = df_step[str(self.feature)].first_valid_index()
                last_idx = df_step[str(self.feature)].last_valid_index()
                if first_idx == None:
                    pass
                else:
                    self.df_feat.loc[self.df_feat['oem'] == str(man), ['feat_released_month']] = df_step.loc[first_idx, 'launch_announced']

                    self.df_feat.loc[self.df_feat['oem'] == str(man), ['feat_removed_month']] = df_step.loc[last_idx, 'launch_announced']

        self._calc_time_dif()

        return self.df_feat


    def _calc_time_dif(self):
        '''
        Encodes time-deltas between each oem release (or removal) of feature, the first oem to do so.
        '''

        self.df_part = self.df_feat.fillna(0)
        self.df_part = self.df_part[self.df_part['feat_released_month']!=0]

        self.df_part.sort_values(['feat_released_month'], inplace=True)
        for n in range(len(self.df_part['feat_released_month'])):
            self.df_part.iloc[n, 2] = self.df_part.iloc[n, 1] - self.df_part.iloc[0, 1]

        self.df_part.sort_values(['feat_removed_month'], inplace=True)
        for n in range(len(self.df_part['feat_removed_month'])):
            self.df_part.iloc[n, 4] = self.df_part.iloc[n, 3] - self.df_part.iloc[0, 3]

        self.df_part.sort_values(['feat_released_month'], inplace=True)


if __name__ == '__main__':

    # Read in the Data after Clean
    '''
    sys.path.append('~/dsi/capstones/cap_3/')

    df_org = pd.read_csv('results/output.csv')

    df_org['launch_announced'] = pd.to_datetime(df_org['launch_announced'])
    df_org['launch_announced'] = df_org['launch_announced'].dt.to_period('M')

    feature_test = 'sensor_altimeter'
    device_test = 'phone'

    feat_altimeter = FEATURE(df_org, feature_test, device_test)
    '''
