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

pd.set_option('display.max_columns', None)


class FEATURE(object):
    
    def __init__(self, df, feature, col_list, device):
        self.df = df
        self.feature = feature
        self.col_list = col_list
        self.df_feat = self._init_df_feat()
        self.device = device

    def _init_df_feat(self):
        man_list = list(self.df['oem'].unique())
        self.df_feat = pd.DataFrame(data=np.full([len(man_list), len(self.col_list)], fill_value = np.nan),
                                    columns = self.col_list)
        self.df_feat['oem'] = man_list

        return self.df_feat

    def choose_device(self):
        if self.device == 'tablet':
            self.df = self.df.loc[self.df['is_tablet'], :]
        elif self.device == 'watch':
            self.df = self.df.loc[self.df['is_watch'], :]
        else:
            df_step = self.df.loc[~self.df.is_tablet, :]
            df_step = df_step.loc[~df_step.is_watch, :]
            self.df = df_step[df_step['network_technology'].notna()]
        return self.df

    def clean_df(self):
        
        return self.df

    
    def _company_release(self):

        pass
    


if __name__ == '__main__':

    # Read in the Data after Clean
    sys.path.append('~/dsi/capstones/cap_3/')

    df_org = pd.read_csv('notebooks/output.csv')
    df_org.drop(columns=['Unnamed: 0'], inplace=True)

    feature_cols = ['oem', 'feat_released_month', 'months_after_release', 
                'feat_removed_month', 'months_before_release']

    feature_test = 'issac'

    issac = FEATURE(df_org, feature_test, feature_cols, 'phone')

