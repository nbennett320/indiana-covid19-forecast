import numpy as np
import pandas as pd
data = pd.read_csv('./data/covid_marion.csv')

print(data['DATE'])

def main():
  dates = data['DATE']
  sum = 0
  sums_list = []
  day = 0
  for i in range(1, len(dates)):
    j = i-1
    if((dates[i] == dates[j]) & (day < 7)):
      sum += data['COVID_COUNT'][j]
    elif((dates[i] != dates[j]) & (day < 7)):
      sum += data['COVID_COUNT'][j]
      day += 1
    else:
      for k in range(0, 7):
        sums_list.append(sum)
      sum = 0
      day = 0
  print(sums_list)
  print(len(sums_list))
  df = pd.DataFrame({'case_count': sums_list})
  df.to_csv(r'./data/avg_cases_by_week.csv')

main()