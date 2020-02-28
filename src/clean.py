# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy as stats
# import re
# import dateparser
# import datetime

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
pd.set_option('display.width', 500)


# CLEANING - Cast NaN/Null
def cast_nan(step_df):

    make_nan = ['No', 'N/A', 'No cellular connectivity', '-']

    df_obj = step_df.select_dtypes(['object'])

    step_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    step_df.replace(to_replace=make_nan, value=np.nan, inplace=True)
    
    return step_df


# CLEANING - Drop Words
useless_drop_words = [
    'launch_status',
    'misc_colors',
    'misc_models',
    'misc_sar_eu',
    'misc_sar',
    'memory',
    'sound_alert_types',
    'features_clock',
    'features_alarm',
    'features_languages',
    'selfie_camera_v1',
    'main_camera',
    'main_camera_quad',
    'selfie_camera_triple',
    'main_camera_v1',
    'main_camera_five'
]

apple_drop_words = [
    'body_keyboard',
    'memory_phonebook',
    'memory_call_records',
    'features_messaging',
    'features_games',
    'features_java'
]

analysis_drop_words = [
    'body_dimensions',
    'memory_internal',
    'battery',
    'battery_talk_time',
    'network_speed',
    'battery_stand-by',
    'tests_performance',
    'tests_display',
    'tests_camera',
    'tests_loudspeaker',
    'tests_audio_quality',
    'tests_battery_life',
    'battery_music_play',
    'selfie_camera',
    'display',
    'features_browser',
    'comms_infrared_port',
    'selfie_camera_dual'
]


# CLEAING - Date Parser
def parse_date(date_str):
    
    # Parsing variables
    years = range(1994, 2017)

    months = ['January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December']

    months_short = ['Jan-', 'Feb-', 'Mar-', 'Apr-', 'May-', 'Jun-', 
                    'Jul-', 'Aug-', 'Sep-', 'Oct-', 'Nov-', 'Dec-']

    months_num = np.linspace(0.0, 1.0, num=12)

    months_int = range(1, 12)

    months_dict = {
        'January':1, 
        'February':2, 
        'March':3, 
        'April':4, 
        'May':5, 
        'June':6, 
        'July':7, 
        'August':8, 
        'September':9, 
        'October':10, 
        'November':11, 
        'December':12
    }

    quarters = ['Q1', '1Q', 'Q2', '2Q', 'Q3', '3Q', 'Q4', '4Q']

    quarters_dict = {
        'Q1': 'February',
        '1Q': 'February',
        'Q2': 'May',
        '2Q': 'May',
        '3Q': 'August',
        'Q3': 'August',
        '4Q': 'November',
        'Q4': 'November'
    }

    # Removes preceding and proceding spaces
    date_str = date_str.strip()
    
    # Changes strings with "?" to "remove"
    if '?' in date_str:
        date_str = 'remove'
    
    # Removes errant release dates
    if '.' in date_str:
        split_str = date_str.split('.')
        date_str = split_str[0]
    
    # Changes Quarter to Middle Month of Quarter
    for key, val in quarters_dict.items():
        if key in date_str:
            split_2 = date_str.split()
            split_2[split_2.index(key)] = val
            date_str = " ".join(split_2)

    # Converts Year Only to Year + ' July'
    if len(date_str) < 5 and len(date_str) > 1:
        date_str += ' July'    
    
    # If date is formated in short form i.e. "Feb-01"
    m_index = 0
    for m in months_short:
        if m in date_str:
            split_3 = date_str.split('-')
            split_3[0] = months[m_index]
            split_3[1] = f'20{split_3[1]}'
            date_str = " ".join(split_3)
        m_index += 1
       
    # Date Parsing str using module dataparser    
    date_tm = dateparser.parse(date_str)
    if isinstance(date_tm, datetime.date):
        date_tm = date_tm - datetime.timedelta(days=date_tm.day-1)
           
    return date_tm


# CLEANING - Execute Date Parser
def refactor_time(df, col='launch_announced'):
    step_df = df[df['launch_announced'].notna()]

    step_df['launch_announced'] = step_df['launch_announced'].apply(parse_date)
    step_df['launch_announced'] = step_df['launch_announced'].dt.to_period('M')
    
    return step_df


# CLEANING - Parse Price
def clean_price(price_value): 

#     price_words = ['About']
#     currencies = ['USD', 'EUR', 'INR']

    if type(price_value)!=str:
        if math.isnan(price_value):
            return np.nan
        
    elif price_value == '-':
        return np.nan
    
    elif price_value == 'About BTC 0.15/ETH 4.78(crypto curr)':
        return 800.0
    
    else:
        price_value = price_value.split(' ')

        if price_value[2] == 'EUR':
            return float(price_value[1]) / 1.2
        elif price_value[2] == 'INR':
            return float(price_value[1]) * 60
        elif price_value[2] == 'USD':
            return float(price_value[1])


# CLEANING - Parse Video 
def clean_video(video_camera_value):
    
    if type(video_camera_value)!=str:
        if math.isnan(video_camera_value):
            return np.nan
    
    if '@' in video_camera_value:
        return 'Yes'
    
    elif 'x' in video_camera_value:
        video_camera_value = re.split('x|@', video_camera_value)
        video_camera_value = int(video_camera_value[0]) * int(video_camera_value[1])
    
    elif video_camera_value != np.nan:
        video_camera_value.replace('w', '')
        video_camera_value = video_camera_value.split('p')[0]
        
        if 'yes' in video_camera_value.lower():
            video_camera_value = 'Yes'
    
    return video_camera_value


'''
# Cleaning - Screen Size

def screen_refactor(s, split_char=' '):
    s = str(s)
    s = s.replace('| -', '')
    s = s.replace('|', '')
    s = s.strip()
    
    
    if isinstance(s, str):
        split_s = s.split(split_char)
        # try 
        return split_s[0]
    else:
        return s
    
period_df['screen_in'] = period_df['display_size'].apply(screen_refactor)
'''


# EXTRACTING - Extract Device Type
tablet_words = ['tab', 'tablet', 'pad', 'book']
watch_words = ['watch', 'gear', 'fit', 'band']

def parse_type(model_string, words_list):

    for check_word in words_list:
        if check_word in model_string.lower():
            return True
        else:
            return False


# EXTRACTION - Features General Function
def extract_features(org_col, data_frame, new_cols, category, 
                     regex_1 = None, regex_2 = None):
    
    for n in new_cols:
        col_str = str(category) + '_' + n.replace(' ', '_')
        data_frame[col_str] = False
    
    idx = 0
    for row in data_frame[str(org_col)]:
        if type(row)!=str:
            if math.isnan(row):
                pass
        else:
            for n in new_cols:
                if n in row.lower():
                    col_str = str(category) + '_' + n.replace(' ', '_')
                    data_frame[col_str].iloc[idx] = True

        idx += 1

# EXTRACTING - Sensors
relevant_sensors = ['accelerometer', 'gyro', 'heart rate', 'fingerprint', 'compass',
                    'proximity', 'barometer', 'spo2', 'iris scanner', 'gesture', 
                    'tempurature', 'altimeter', 'infrared face recognition'
                   ]


'''
# EXTRACTING - FINDING UNIQUE PHONE MODELS (NOT COMPLETE)

def unique_models(_list, split_1 = None, split_2 = None):
#     _list = model_list
    unique_list = []
    
    if split_1 is None and split_2 is None:
        for n in _list:
            unique_list.append(n)
            unique_list = is_unique(unique_list)

    elif split_1:
        for n in _list:
            n_dict = {}
            split_atr = n.split(split_1)
            n_dict[split_atr[0]] = []
            n_dict[split_atr[0]].extend(split_atr[1:])
            unique_list.append(n_dict)
    
    return unique_list

'''

if __name__=='__main__':

    df_org = pd.read_csv('data/gsm.csv')
    df = df_org.copy()

    df = cast_nan(df)

    df.drop(columns=useless_drop_words, inplace=True)
    df.drop(columns=apple_drop_words, inplace=True)
    df.drop(columns=analysis_drop_words, inplace=True)

    df = refactor_time(df)

    df['misc_price'].apply(lambda x: clean_price(x))

    df['main_camera_video'] = df['main_camera_video'].apply(clean_video)

    df['is_tablet'] = df['model'].apply(lambda x: parse_type(x, tablet_words))
    df['is_watch'] = df['model'].apply(lambda x: parse_type(x, watch_words))

    extract_features(org_col = 'features_sensors', 
                 data_frame = df, 
                 new_cols = relevant_sensors, 
                 category = 'sensor'
                )

    df.to_csv('output.csv')