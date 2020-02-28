import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import sys, os
sys.path.append('~/dsi/capstones/cap_3/')

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DateType
import pyspark as ps

import scipy as stats
import re
import dateparser
import datetime
import math

import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols

from common import is_bool_dtype

pd.set_option('display.max_columns', None)


class FEATURE(object):
    
    def __init__(self, df, feature, device):
        self.df = df
        self.feature = feature
        self.device = device
        self.col_list = ['oem', 'feat_released_month', 'months_after_release', 
                                'feat_removed_month', 'months_before_release']

        self.df_device = None
        self.df_feat = self._company_release()
        

    def _clean_df(self):
        if 'Unnamed: 0' in self.df.columns:
           self.df.drop(columns=['Unnamed: 0'], inplace=True) 

        return self.df


    def _choose_device(self):
        if self.device == 'tablet':
            self.df = self.df.loc[self.df['is_tablet'], :]
        elif self.device == 'watch':
            self.df = self.df.loc[self.df['is_watch'], :]
        else:
            df_step = self.df.loc[~self.df.is_tablet, :]
            df_step = df_step.loc[~df_step.is_watch, :]
            self.df = df_step[df_step['network_technology'].notna()]
        return self.df
    

    def _init_df_feat(self):
        man_list = list(self.df['oem'].unique())
        self.df_feat = pd.DataFrame(data=np.full([len(man_list), len(self.col_list)], 
                                                fill_value = np.nan),
                                            columns = self.col_list)
        self.df_feat['oem'] = man_list

        return self.df_feat


    def _company_release(self):
        
        self._clean_df()
        self._choose_device()
        self._init_df_feat()

        man_list = list(self.df['oem'].unique())
        
        for man in man_list:
            df_step = self.df_device[self.df_device['oem'] == str(man)].sort_values(by='launch_announced')
            
            if is_bool_dtype(df_step[str(self.feature)]):
                first_idx = df_step[df_step[str(self.feature)]==True].first_valid_index()
                last_idx = df_step[df_step[str(self.feature)]==True].last_valid_index()
                if first_idx == None:
                    pass
                else:
                    self.df_feat.groupby('oem').loc[str(man), 'feat_released_month'] = df_step.loc[first_idx, 'launch_announced']

                    self.df_feat.groupby('oem').loc[str(man), 'feat_removed_month'] = df_step.loc[last_idx, 'launch_announced']

            else:
                first_idx = df_step[str(self.feature)].first_valid_index()
                last_idx = df_step[str(self.feature)].last_valid_index()
                if first_idx == None:
                    pass
                else:
                    self.df_feat.groupby('oem').loc[str(man), 'feat_released_month'] = df_step.loc[first_idx, 'launch_announced']

                    self.df_feat.groupby('oem').loc[str(man), 'feat_removed_month'] = df_step.loc[last_idx, 'launch_announced']

        return self.df_feat

        
        
if __name__ == '__main__':

    # Read in the Data after Clean
    sys.path.append('~/dsi/capstones/cap_3/')

    df_org = pd.read_csv('notebooks/output.csv')
    df_org.drop(columns=['Unnamed: 0'], inplace=True)

    feature_cols = ['oem', 'feat_released_month', 'months_after_release', 
                    'feat_removed_month', 'months_before_release']

    feature_test = 'sensor_altimeter'

    issac = FEATURE(df_org, feature_test, 'phone')

