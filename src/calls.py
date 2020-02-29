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
from clean import parse_date
from clean import refactor_time
from feature import FEATURE
import clean

# from operator import attrgetter
# df['duration_dataset'] = (
#     df['date_1'].dt.to_period('M') -
#     df['date_2'].dt.to_period('M')).apply(attrgetter('n'))


class CALLS(object):

    def __init__(self, df, feature, device):
        self.df = df
        self.feature = feature
        self.device = device
        self.col_list = ['oem', 'feat_released_month', 'months_after_release', 
                                'feat_removed_month', 'months_after_removal']

        # self.df_device = None
        self.df_part = pd.DataFrame()
        self.df_feat = self._company_release()