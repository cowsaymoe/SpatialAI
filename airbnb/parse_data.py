import sys
import os
import pandas as pd

# load raw airbnb listings data in CSV format into dataframe
# https://public.opendatasoft.com/explore/dataset/airbnb-listings/map/?disjunctive.host_verifications&disjunctive.amenities&disjunctive.features&geofilter.polygon=(42.22785637852162,-71.24153137207031),(42.41519403261846,-71.24153137207031),(42.41519403261846,-70.9881591796875),(42.22785637852162,-70.9881591796875),(42.22785637852162,-71.24153137207031)&location=11,42.31286,-71.08601&basemap=jawg.light
df = pd.read_csv(r'data/airbnb-listings.csv', encoding = 'unicode_escape')
# if you want to use excel instead
# df = pd.read_excel (r'airbnb-listings.xls.xlsx', sheet_name='Sheet')
df