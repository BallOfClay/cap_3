



# ANALYSIS - GENERAL TRENDS

# Average Price Over Time
fig = plt.figure(figsize=(12,4))

ax1 = period_df.groupby('announced').mean()['approx_price_EUR'].plot(
    xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')], ylim=[0, 750])

ax1.set_ylabel('Average Price (EUR)')
ax1.set_xlabel('Month Announced')


# Average Screen Size Over Time
fig2 = plt.figure(figsize=(12,4))

ax2 = period_df.groupby('announced').mean()['screen_in'].plot(
    xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')])

ax2.set_ylabel('Average Price (EUR)')
ax2.set_xlabel('Month Announced')

# Average Battery Capacity Over Time
ax3 = period_df.groupby('announced').mean()['screen_in'].plot(
    xlim=[pd.Timestamp('2005-08-01'), pd.Timestamp('2017-10-01')])

ax3.set_ylabel('Average Price (EUR)')
ax3.set_xlabel('Month Announced')


# ANALYSIS - SPECIFIC FEATURES

df_feature = period_df[['bluetooth', 'announced']]
    
date_max = df_feature['announced'].max()
date_min = df_feature['announced'].min()

month_period = month_diff(date_max, date_min)
day_period = date_max - date_min

date_counts = df_feature.groupby([df_feature['announced'].dt.year, df_feature['announced'].dt.month]).count()


