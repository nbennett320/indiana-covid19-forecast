import numpy as np
import pandas as pd
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv', index_col=8)
df = pd.DataFrame(mobility_location_data)

def main():
  sorted_df = df.sort_values(by='date')
  sorted_df.to_csv(r'./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized_sorted_2.csv')

main()