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
import json

import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols

from common import is_bool_dtype
from clean import parse_date
from clean import refactor_time
from feature import FEATURE
import clean

from collections import Counter

from operator import attrgetter
# df['duration_dataset'] = (
#     df['date_1'].dt.to_period('M') -
#     df['date_2'].dt.to_period('M')).apply(attrgetter('n'))


# def time_delta_counter(df_feat, is_apple=False):
    
#     if is_apple:




if __name__=='__main__':


    # Read in the Data after Clean
    sys.path.append('~/dsi/capstones/cap_3/')

    df_org = pd.read_csv('notebooks/output.csv')
    df_org = refactor_time(df_org)
    df_org = df_org.rename(columns = {'sound_3.5mm_jack':'sound_3_5mm_jack'})

    df_org = df_org.loc[df_org['launch_announced'] >= '2006']

    current_feat_list = [
        'network_gprs',
        'network_edge',
        'memory_card_slot',
        'main_camera_video',
        'selfie_camera_video',
        'sound_3_5mm_jack',
        'comms_radio',
        'sensor_accelerometer',
        'sensor_gyro',
        'sensor_heart_rate',
        'sensor_fingerprint',
        'sensor_compass',
        'sensor_proximity',
        'sensor_barometer',
        'sensor_spo2',
        'sensor_iris_scanner',
        'sensor_gesture',
        'sensor_tempurature',
        'sensor_altimeter',
        'sensor_infrared_face_recognition',
        'comms_nfc',
        'main_camera_dual',
        'main_camera_triple'
    ]

    chosen_device = 'phone'
    # breakpoint()
    # for col in current_feat_list:
    #     # f'col'= FEATURE(df_org, col, 'phone')
    #     feat_name = exec("%s = %d" % (col,2))
    #     print(feat_name)
    #     feat_name = FEATURE(df_org, col, chosen_device)

    feat_obj_lst = [FEATURE(df_org, n, chosen_device) for n in current_feat_list]
    
    df_dict = {}
    # breakpoint()
    for idx, feat in enumerate(current_feat_list):
        df_dict[feat] = feat_obj_lst[idx].df_part
    # for obj in objs:
    #     other_object.add(obj)


    '''
    network_gprs = FEATURE(df_org, 'network_gprs', 'phone')

    network_edge = FEATURE(df_org, 'network_edge', 'phone')

    memory_card_slot = FEATURE(df_org, 'memory_card_slot', 'phone')

    main_camera_video = FEATURE(df_org, 'main_camera_video', 'phone')

    selfie_camera_video = FEATURE(df_org, 'selfie_camera_video', 'phone')

    sound_3_5mm_jack = FEATURE(df_org, 'sound_3_5mm_jack', 'phone')

    comms_radio = FEATURE(df_org, 'comms_radio', 'phone')

    sensor_accelerometer = FEATURE(df_org, 'sensor_accelerometer', 'phone')
    sensor_gyro = FEATURE(df_org, 'sensor_gyro', 'phone')
    sensor_heart_rate = FEATURE(df_org, 'sensor_heart_rate', 'phone')
    sensor_fingerprint = FEATURE(df_org, 'sensor_fingerprint', 'phone')
    sensor_compass = FEATURE(df_org, 'sensor_compass', 'phone')
    sensor_proximity = FEATURE(df_org, 'sensor_proximity', 'phone')
    sensor_barometer = FEATURE(df_org, 'sensor_barometer', 'phone')
    sensor_spo2 = FEATURE(df_org, 'sensor_spo2', 'phone')
    sensor_iris_scanner = FEATURE(df_org, 'sensor_iris_scanner', 'phone')
    sensor_gesture = FEATURE(df_org, 'sensor_gesture', 'phone')
    sensor_tempurature = FEATURE(df_org, 'sensor_tempurature', 'phone')
    sensor_altimeter = FEATURE(df_org, 'sensor_altimeter', 'phone')
    sensor_infrared_face_recognition = FEATURE(df_org, 'sensor_infrared_face_recognition', 'phone')

    comms_nfc = FEATURE(df_org, 'comms_nfc', 'phone')

    main_camera_dual = FEATURE(df_org, 'main_camera_dual', 'phone')

    main_camera_triple = FEATURE(df_org, 'main_camera_triple', 'phone')

    '''


    n_bins = 10
    dict_len = 250
    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes.flatten()

    other_months_after_release = dict(zip(range(0, dict_len), [0]*dict_len))
    other_months_after_removal = dict(zip(range(0, dict_len), [0]*dict_len))

    apple_months_after_release = dict(zip(range(0, dict_len), [0]*dict_len))
    apple_months_after_removal = dict(zip(range(0, dict_len), [0]*dict_len))
    
    # breakpoint()
    for key_feat, df_part_feat in df_dict.items():
        # n_df_part = feat_obj_lst[eval(n)].df_part
        # n_df_part = eval(n).df_part
        df_part_feat['months_after_release'] = df_part_feat['months_after_release'].apply(attrgetter('n'))
        df_part_feat['months_after_removal'] = df_part_feat['months_after_removal'].apply(attrgetter('n'))
        # breakpoint()
        if df_part_feat['oem'].str.contains('Apple').any():
            # apple_months_after_release[df_part_feat[df_part_feat['oem']=='Apple']['months_after_release']] += 1
            # apple_months_after_removal[df_part_feat[df_part_feat['oem']=='Apple']['months_before_removal']] += 1
            # df_part_feat = df_part_feat[df_part_feat[df_part_feat['oem']!='Apple']]
            
            # breakpoint()
            n_months = df_part_feat.loc[df_part_feat['oem']=='Apple', 'months_after_release']
            apple_months_after_release[int(n_months)] += 1
            
            n_months = df_part_feat.loc[df_part_feat['oem']=='Apple', 'months_after_removal']
            apple_months_after_removal[int(n_months)] += 1
            
            df_part_feat = df_part_feat[df_part_feat['oem']!='Apple']
            # breakpoint()
            for n_month in df_part_feat['months_after_release']:
                # if n_df_part['months_after_release']
                # breakpoint()
                # n_month = n_month.dt.astype(int)
                other_months_after_release[int(n_month)] += 1

            for n_month in df_part_feat['months_after_removal']:
                # n_month = n_month.apply(attrgetter('n'))
                other_months_after_removal[int(n_month)] += 1


    # ax0.hist(x_multi, n_bins, histtype='bar')
    # ax0.set_title('different sample sizes')
    

    with open('other_months_after_release_200228.json', 'w') as f:
        json.dump(other_months_after_release, f)

    with open('other_months_after_removal_200228.json', 'w') as f:
        json.dump(other_months_after_removal, f)

    with open('apple_months_after_release_200228.json', 'w') as f:
        json.dump(apple_months_after_release, f)

    with open('apple_months_after_removal_200228.json', 'w') as f:
        json.dump(apple_months_after_removal, f)

    
    apple_release_array = []
    other_release_array = []

    apple_removal_array = []
    other_removal_array = []

    for key, val in apple_months_after_release.items():
        if val != 0:
            apple_release_array.append([key]*val)

    for key, val in other_months_after_release.items():
        other_release_array.append([key]*val)

    for key, val in apple_months_after_removal.items():
        apple_removal_array.append([key]*val)

    for key, val in other_months_after_removal.items():
        other_removal_array.append([key]*val)


    '''
    fig = plt.figure(figsize=(12,4))

    ax1 = df_org.groupby('launch_announced').mean()['misc_price'].plot(
        xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')], ylim=[0, 750])

    ax1.set_ylabel('Average Price (USD)')
    ax1.set_xlabel('Month Announced')


    # Average Screen Size Over Time
    fig2 = plt.figure(figsize=(12,4))

    ax2 = df_org.groupby('launch_announced').mean()['screen_in'].plot(
        xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')])

    ax2.set_ylabel('Screen Size (in)')
    ax2.set_xlabel('Month Announced')

    # Average Battery Capacity Over Time
    ax3 = df_org.groupby('launch_announced').mean()['battery'].plot(
        xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')])

    ax3.set_ylabel('battery capacity')
    ax3.set_xlabel('Month Announced')
    '''


