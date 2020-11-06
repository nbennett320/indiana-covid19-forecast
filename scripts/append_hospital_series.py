import numpy as np
import pandas as pd
import re, numbers, decimal, math
county_level_data = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_6.csv', index_col=0)
bed_vent_data = pd.read_csv('./data/covid_report_bedvent_date.csv', index_col=0)
c_df = pd.DataFrame(county_level_data)
bv_df = pd.DataFrame(bed_vent_data)

def main():
  for col in bv_df.columns:
    c_df.insert(len(c_df.columns), col, 0)
  print(c_df)
  
  for i, j in bv_df.iterrows():
    print('i', i)
    for x, y in j.iteritems():
      try:
        c_df.at[i, x] = y
        # print('val', c_df.at[i, x])
      except:
        break
      # finally:
        # print('i', i)
        # print('j', j)
        # print('x', x)
        # print('y', y)
  print(c_df)
  c_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted_7.csv')

main()