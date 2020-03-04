import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

import csv
import sys, os
sys.path.append('~/dsi/capstones/cap_3/')

import scipy
import re
import dateparser
import datetime
import math
import json
import itertools

# import researchpy as rp
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

from common import is_bool_dtype
from clean import parse_date
from clean import refactor_time
from feature import FEATURE
import clean

from collections import Counter
from operator import attrgetter


if __name__=='__main__':

    # Read in the Data after Clean
    sys.path.append('~/dsi/capstones/cap_3/')
    df_org = pd.read_csv('results/output.csv')

    df_org['launch_announced'] = pd.to_datetime(df_org['launch_announced'])
    df_org['launch_announced'] = df_org['launch_announced'].dt.to_period('M')

    df_org = df_org.rename(columns = {'sound_3.5mm_jack':'sound_3_5mm_jack'})  # Rename 3.5 mm jack

    df_org = df_org.loc[df_org['launch_announced'].dt.year >= 2006]  #

    oems_less_than_four = ['Noenode', 'Thuraya', 'Razer', 'Fujitsu Siemens', 'Benefon', 'XCute', 
                            'Jolla', 'Nvidia', 'Qtek']
    df_org = df_org[~df_org['oem'].isin(oems_less_than_four)]


    current_feat_list = [
        'network_gprs',
        'network_edge',
        'memory_card_slot',
        'main_camera_video',
        'selfie_camera_video',
        'sound_3_5mm_jack',
        'comms_gps',
        'comms_radio',
        'network_3g_bands',
        'network_4g_bands',
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
        'main_camera_triple',
        'main_camera_quad',
        'body_water_resistant',
        'body_waterproof',
        'body_pay',
        'body_stylus',
        'body_kickstand',
        'body_flashlight',
        'display_type_LCD',
        'display_type_OLED',
        'display_type_AMOLED',
        'display_type_TFT',
        'display_type_STN',
        'display_type_CSTN',
        'display_type_ASV',
        'display_type_IPS',
        'display_type_resistive',
        'display_type_capacitive',
        'display_type_touchscreen',
        'platform_cpu_octa-core',
        'platform_cpu_hexa-core',
        'platform_cpu_quad-core',
        'platform_cpu_dual-core',
        # 'comms_wlan_a',
        # 'comms_wlan_b',
        'comms_wlan_g',
        'comms_wlan_i',
        'comms_wlan_n',
        'comms_wlan_ac',
        'comms_wlan_ax',
        'comms_wlan_dual-band',
        'comms_wlan_hotspot',
        'comms_wlan_DLNA',
        'comms_wlan_Wi-Fi_Direct',
        'comms_bluetooth_2.1',
        'comms_bluetooth_2.2',
        'comms_bluetooth_3.0',
        'comms_bluetooth_3.1',
        'comms_bluetooth_4.0',
        'comms_bluetooth_4.1',
        'comms_bluetooth_4.2',
        'comms_bluetooth_5.0',
        'comms_bluetooth_5.1',
        'comms_bluetooth_A2DP',
        'comms_bluetooth_EDR',
        'comms_bluetooth_LE',
        'comms_bluetooth_aptX',
        'battery_removable',
        'battery_li-ion',
        'battery_li-po',
        'main_camera_features_flash',
        'main_camera_features_HDR',
        'main_camera_features_panorama',
        'sound_active_noise_cancelation',
        'sound_dedicated_mic',
        'sound_HDR',
        'selfie_camera_features_flash',
        'selfie_camera_features_HDR',
        'battery_charging_fast',
        'battery_charging_wireless',
        'battery_charging_reverse',
        'display_home_button',
        'display_3D',
        'body_build_plastic_back',
        'body_build_glass_back',
        'body_build_ceramic_back',
        'network_technology_GSM',
        'network_technology_CDMA',
        'network_technology_HSPA',
        'network_technology_EDVO',
        'network_technology_LTE',
        'network_technology_UMTS',
        'network_TD-SCDMA',
        'network_HSDPA',
        'sound_loudspeaker_stereo',
        'platform_cpu_single-core',
        'camera_feature_wide',
        'camera_feature_ultrawide',
        'camera_feature_telephoto',
        'camera_feature_zoom',
        'camera_feature_depth',
        'camera_feature_laser',
        'camera_feature_ois',	
        'camera_feature_pdaf'
    ]

    chosen_device = 'phone'
    
    feat_obj_lst = [FEATURE(df_org, n, chosen_device) for n in current_feat_list]

    # Creates Phone specific dataframe for review
    df_phone = df_org.loc[(df_org['is_tablet'] == False) & (df_org['is_watch'] == False)]
    df_phone.to_csv('results/selected_df.csv')

    df_dict = {}
    for idx, feat in enumerate(current_feat_list):
        df_dict[feat] = feat_obj_lst[idx].df_part


    # Creates Dictionaries Counters to count time-deltas for all features
    n_bins = 10
    dict_len = 200

    other_months_after_release = dict(zip(range(0, dict_len), [0]*dict_len))
    other_months_after_removal = dict(zip(range(0, dict_len), [0]*dict_len))

    apple_months_after_release = dict(zip(range(0, dict_len), [0]*dict_len))
    apple_months_after_removal = dict(zip(range(0, dict_len), [0]*dict_len))
    
    for key_feat, df_part_feat in df_dict.items():

        df_part_feat['months_after_release'] = df_part_feat['months_after_release'].apply(attrgetter('n'))
        df_part_feat['months_after_removal'] = df_part_feat['months_after_removal'].apply(attrgetter('n'))

        if df_part_feat['oem'].str.contains('Apple').any():

            n_months = df_part_feat.loc[df_part_feat['oem']=='Apple', 'months_after_release']
            apple_months_after_release[int(n_months)] += 1
            
            n_months = df_part_feat.loc[df_part_feat['oem']=='Apple', 'months_after_removal']
            apple_months_after_removal[int(n_months)] += 1
            
            df_part_feat = df_part_feat[df_part_feat['oem']!='Apple']

            for n_month in df_part_feat['months_after_release']:
                other_months_after_release[int(n_month)] += 1

            for n_month in df_part_feat['months_after_removal']:
                other_months_after_removal[int(n_month)] += 1

    with open('results/other_months_after_release_dict.json', 'w') as f:
        json.dump(other_months_after_release, f)

    with open('results/other_months_after_removal_dict.json', 'w') as f:
        json.dump(other_months_after_removal, f)

    with open('results/apple_months_after_release_dict.json', 'w') as f:
        json.dump(apple_months_after_release, f)

    with open('results/apple_months_after_removal_dict.json', 'w') as f:
        json.dump(apple_months_after_removal, f)

    
    # Creates flattened arrays from Dictionary Counters
    apple_release_array = []
    other_release_array = []

    apple_removal_array = []
    other_removal_array = []

    for key, val in apple_months_after_release.items():
        if val != 0:
            apple_release_array.append([key]*val)

    for key, val in other_months_after_release.items():
        if val != 0:
            other_release_array.append([key]*val)

    for key, val in apple_months_after_removal.items():
        if val != 0:
            apple_removal_array.append([key]*val)

    for key, val in other_months_after_removal.items():
        if val != 0:
            other_removal_array.append([key]*val)

    apple_release_array = list(itertools.chain(*apple_release_array))
    other_release_array = list(itertools.chain(*other_release_array))
    apple_removal_array = list(itertools.chain(*apple_removal_array))
    other_removal_array = list(itertools.chain(*other_removal_array))

    removal_bound = max(other_removal_array)
    

    # Determine Mean, Variance and Standard Deviations of Flattened Arrays
    a_release_mean = np.mean(apple_release_array)
    o_release_mean = np.mean(other_release_array)
    a_removal_mean = np.mean(apple_removal_array)
    o_removal_mean = np.mean(other_removal_array)

    a_release_var = np.var(apple_release_array)
    o_release_var = np.var(other_release_array)
    a_removal_var = np.var(apple_removal_array)
    o_removal_var = np.var(other_removal_array)

    a_release_std = np.std(apple_release_array)
    o_release_std = np.std(other_release_array)
    a_removal_std = np.std(apple_removal_array)
    o_removal_std = np.std(other_removal_array)

    release_test = scipy.stats.ttest_ind(apple_release_array, other_release_array, equal_var=False)
    removal_test = scipy.stats.ttest_ind(apple_removal_array, other_removal_array, equal_var=False)

    with open('results/release_ttest.json', 'w') as f:
        json.dump(release_test, f)

    with open('results/removal_ttest.json', 'w') as f:
        json.dump(removal_test, f)


    # Creates Plot comparing Apple Release behavior versus the competition
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.hist([apple_release_array, other_release_array], bins=20)

    ax1.set_xlabel('Time-Delta between each OEM Release & First Release (months)', size = 18)
    ax1.set_ylabel('Counts of Time-Deltas', size = 18)
    ax1.set_title('Months Between First Release & Other Manufacturer Adoption of All Features', size = 24)
    ax1.axvline(a_release_mean, color='blue', label='apple mean', linestyle='--', dashes=(5, 5)) 
    ax1.axvline(o_release_mean, color='green', label='other mean', linestyle='--', dashes=(5, 10)) 
    ax1.legend(loc='upper right', framealpha = 0.5)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14) 

    plt.show()
    plt.savefig('results/features_release.png')

    # Creates Plot comparing Apple Removal behavior versus the competition
    fig2, ax2 = plt.subplots(figsize=(12,8))
    ax2.hist([apple_removal_array, other_removal_array], bins=20)

    ax2.set_xlabel('Time-Delta between each OEM Removal & First Removal (months)', size = 18)
    ax2.set_ylabel('Counts of Time-Deltas', size = 18)
    ax2.set_title('Months Between First Removal & Other Manufacturer Removal of All Features', size = 24)
    ax2.axvline(a_removal_mean, color='blue', label='apple mean', linestyle='--', dashes=(5, 5)) 
    ax2.axvline(o_removal_mean, color='green', label='other mean', linestyle='--', dashes=(5, 10)) 
    ax2.legend(loc='upper right', framealpha = 0.5)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14) 

    plt.show()
    plt.savefig('results/features_removal.png')


    # Creates Plot showing mean price over time
    fig3, ax3 = plt.subplots(figsize=(12,4))
    ax3 = df_org.groupby('launch_announced').mean()['misc_price'].plot(
        xlim=[pd.Timestamp('2006-01-01'), pd.Timestamp('2019-04-01')])

    ax3.set_ylabel('Average Price (USD)')
    ax3.set_xlabel('Month Announced')
    ax3.set_title('Average Phone Price per Month')

    plt.show()
    plt.savefig('results/price_over_time.png')


    # Creates Plot showing mean display size over time
    fig4, ax4 = plt.subplots(figsize=(12,4))
    ax4 = df_org.groupby('launch_announced').mean()['display_size'].plot(
        xlim=[pd.Timestamp('2006-01-01'), pd.Timestamp('2019-04-01')])

    ax4.set_ylabel('Average Screen Size (in)')
    ax4.set_xlabel('Month Announced')
    ax4.set_title('Average Display Size per Month')

    plt.show()
    plt.savefig('results/screen_size_over_time.png')


    # Creates Plot showing aggregate number of phones released over time
    fig5, ax5 = plt.subplots(figsize=(12,4))
    ax5 = df_org.groupby('launch_announced').agg({'model':'count'}).plot(
        xlim=[pd.Timestamp('2006-01-01'), pd.Timestamp('2019-04-01')])

    ax5.set_ylabel('# Phones Announced')
    ax5.set_xlabel('Month Announced')
    ax5.set_title('Phones Announced per Month')

    plt.show()
    plt.savefig('results/phone_per_month.png')


    '''
    # Attempts to create plot showing Apple release behavior versus the competition on two axes
    fig6, ax6 = plt.subplots()
    ax6.hist(apple_release_array, color='yellow')

    ax6.set_ylabel('Apple Counts (Time-Deltas)')
    ax6.set_xlabel('(OEM Feature Release - First Release) per Feature per Company (time-delta months)') 
    ax6.set_title('Months Between First Release & Other Manufacturer Adoption of All Features')
    ax6.axvline(a_release_mean, color='green', label='apple mean dist.') 

    # n, bins, patches = ax6.hist(apple_release_array, other_release_array])
    # ax6.xlabel('(Apple Feature Release - First Release) per Feature (time-delta months)')
    # ax7.xlabel('(Other Feature Release - First Release) per Feature (time-delta months)')

    ax7 = ax6.twinx()
    ax6.hist(other_release_array, color='blue')
    ax7.set_ylabel('Other Counts (Time-Deltas)')
    ax7.axvline(o_release_mean, color='red', label='other mean dist.') 
    ax6.legend(loc='upper right')

    plt.show()
    plt.savefig('results/release_distribuitions.png')
    '''

    
    # Attempts to create plot showing Apple release behavior versus the competition on two axes
    fig6, ax6 = plt.subplots(figsize=(12,6))
    ax7 = ax6.twinx()
    ax6.hist([apple_release_array, other_release_array])
    # n, bins, patches = ax6.hist(apple_release_array, other_release_array])

    ax6.set_xlabel('(Time between Apple Feature Release - First Release) per Feature (time-delta months)')
    # ax7.xlabel('(Other Feature Release - First Release) per Feature (time-delta months)')
    ax6.set_ylabel('Apple Counts (Time-Deltas)')
    ax6.set_title('Months Between First Release & Other Manufacturer Adoption of All Features')
    ax6.axvline(a_release_mean, color='green', label='apple mean')
    ax6.legend(loc='upper right')
    ax7.set_ylabel('Other Counts (Time-Deltas)')
    ax7.axvline(o_release_mean, color='red', label='other mean') 
   
    plt.show()
    plt.savefig('results/double_histpng')


    '''
    # Creates Plot showing average price over time
    fig8, ax8
    ax8 = df_org.groupby('launch_announced').mean()['battery'].plot(
        xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')])

    ax8.set_ylabel('battery capacity (mAh)')
    ax8.set_xlabel('Month Announced')
    ax8.set_title('Average Display Size per Month')

    plt.show()
    plt.savefig('results/screen_size_over_time.png')
    '''


