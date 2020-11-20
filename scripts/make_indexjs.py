import pandas as pd

county_data = pd.read_csv('./data/indiana_counties.csv')
cdf = pd.DataFrame(county_data)
file = open('./frontend/src/data/index.js', 'w')
for i in cdf['county_name']:
  county = str(i).replace(' ','').replace('.','')
  name = 'model_prediction_' + county + '_covid_count.json'
  l = 'export { default as ' + county + 'Data } from \'./' + name + '\'\n'
  file.write(l)
file.close()