import sys
import os
import pandas as pd

# load raw airbnb listings data in CSV format into dataframe
# https://public.opendatasoft.com/explore/dataset/airbnb-listings/map/?disjunctive.host_verifications&disjunctive.amenities&disjunctive.features&geofilter.polygon=(42.22785637852162,-71.24153137207031),(42.41519403261846,-71.24153137207031),(42.41519403261846,-70.9881591796875),(42.22785637852162,-70.9881591796875),(42.22785637852162,-71.24153137207031)&location=11,42.31286,-71.08601&basemap=jawg.light
df = pd.read_csv(r'data/airbnb-listings.csv', encoding = 'unicode_escape')
# if you want to use excel instead
# df = pd.read_excel (r'airbnb-listings.xls.xlsx', sheet_name='Sheet')

print('Unparsed dataframe loaded:')
print(df)

print('Data attributes:')
print(df.columns)

# create a new dataframe with only the necessary columns
df_new = df[['Host Response Rate',
       'Host Acceptance Rate', 'Host Listings Count', 'Bathrooms', 'Bedrooms',
       'Beds','Square Feet', 'Price', 'Weekly Price',
       'Monthly Price', 'Number of Reviews', 'Review Scores Rating', 'Review Scores Accuracy',
       'Review Scores Cleanliness', 'Review Scores Checkin',
       'Review Scores Communication', 'Review Scores Location',
       'Review Scores Value', 'Reviews per Month', 'Geolocation']]

print('New dataframe with only necessary columns:')
print(df_new)

# search for NaN/null values in the dataframe
print(df_new.isnull())

# count nan values
att_nan_count = df_new.isnull().sum(axis = 0)
print(att_nan_count)

(df_new=='NaN')
(df_new==1)

# transpose nan count to frame
att_nan_count_df = att_nan_count.to_frame().transpose()
print(att_nan_count_df)

# find attributes with less than 500 nan values and remove attribtues with more than 500 nan values
(att_nan_count_df<500)
(att_nan_count_df<500).values[0]

att_nan_count_df.loc[:, (att_nan_count_df<500).values[0]]

att_subset = att_nan_count_df.loc[:, (att_nan_count_df<500).values[0]]

# attributes that are left
print('Attributes that are left:')
print(att_subset.keys())

df_subset = df_new[att_subset.keys()]

print(df_subset)

df_subset_row = df_subset.dropna()
df_subset_row.reset_index(drop=True)
print(df_subset_row)

# save processed dataframe to csv
print('Saving processed dataframe to csv...')
df_subset_row.to_csv('data/airbnb-listings-processed.csv', sep=',')