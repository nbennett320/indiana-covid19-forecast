import pandas as pd
import numpy as np
import requests
import os, sys, re
from zipfile import ZipFile
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# imported data files
county_level_data = pd.read_csv('./data/indiana_county_level_data.csv')
county_level_data_df = pd.DataFrame(county_level_data)

urls = {
  'jh_cases': "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
  'jh_deaths': "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",
  'google_regional_mobility_report': "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip",
  'apple_covid_dashboard': "https://covid19.apple.com/mobility",
  'apple_vehicle_mobility_report': "",
  'indiana_hospital_vent_data': "https://hub.mph.in.gov/dataset/4d31808a-85da-4a48-9a76-a273e0beadb3/resource/0c00f7b6-05b0-4ebe-8722-ccf33e1a314f/download/covid_report_bedvent_date.xlsx",
  'indiana_county_wide_test_case_death_trends': "https://hub.mph.in.gov/dataset/bd08cdd3-9ab1-4d70-b933-41f9ef7b809d/resource/afaa225d-ac4e-4e80-9190-f6800c366b58/download/covid_report_county_date.xlsx",
  'indiana_covid_demographics_by_county_and_district': "https://hub.mph.in.gov/dataset/07e12c46-eb38-43cf-b9e1-46a9c305beaa/resource/9ae4b185-b81d-40d5-aee2-f0e30405c162/download/covid_report_demographics_county_district.xlsx",
  'indiana_covid_deaths_by_date_by_age_group': "https://hub.mph.in.gov/dataset/6bcfb11c-6b9e-44b2-be7f-a2910d28949a/resource/7661f008-81b5-4ff2-8e46-f59ad5aad456/download/covid_report_death_date_agegrp.xlsx",
  'indiana_covid_cases_by_school': "https://hub.mph.in.gov/dataset/61c058c1-abfc-48fb-9bd7-bbf052fe61d6/resource/39239f34-11ff-4dfc-9b9a-a408b0399458/download/covid_report_cases_by_school.xlsx",
}

# initialize data
dfs = {
  'jh_cases': None,
  'jh_deaths': None,
  'google_regional_mobility_report': None,
  'apple_covid_dashboard': None,
  'apple_vehicle_mobility_report': None,
  'indiana_hospital_vent_data': None,
  'indiana_county_wide_test_case_death_trends': None,
  'indiana_covid_demographics_by_county_and_district': None,
  'indiana_covid_deaths_by_date_by_age_group': None,
  'indiana_covid_cases_by_school': None
}

# indiana confirmed totals dataset column names
indiana_confirmed_totals_cols = [
    'date', 
    'county', 
    'state', 
    'country', 
    'combined_key', 
    'lat', 
    'long', 
    'fips', 
    'cases', 
    'deaths',
    'avg_cases_last_week',
    'avg_deaths_last_week',
    'avg_cases_last_2_weeks',
    'avg_deaths_last_2_week',
    'std_cases_last_week',
    'std_deaths_last_week',
    'std_cases_last_2_weeks',
    'std_deaths_last_2_weeks',
    'median_gross_rent',
    'average_household_size',
    'building_permits_number',
    'percent_households_with_computer',
    'percent_households_with_broadband_internet',
    'dollars_per_capita_income_in_past_12_months_2018',
    'population_per_square_mile',
    'median_household_income',
    '2019_population_estimate',
    '2010_population',
    'percent_housing_units_in_multi_unit_structures',
    'percent_under_65_without_health_insurance',
    'percent_under_65_with_disability'
  ]

def fetch_apple_mobility_report_url(i=0):
  if i == 4:
    print("failed to fetch apple mobility report url 5 times in a row... did something break..?\ncanceled dataset update.")
    sys.exit(1)
  else:
    if i > 0:
      print(f"failed to fetch apple mobility report url.\nattempting again... ({i}/4)")
    browser = webdriver.Chrome('/snap/bin/chromium.chromedriver')
    browser.get(url=urls['apple_covid_dashboard'])
    wait = WebDriverWait(browser, 20)
    wait.until(EC.visibility_of_any_elements_located((By.CLASS_NAME, 'download-button-container')))
    soup = BeautifulSoup(browser.page_source, 'lxml')
    dl_card = soup.find('div', attrs={'id': 'download-card'})
    dl_btn = dl_card.find('div', attrs={'class': 'download-button-container'})
    a = dl_btn.find('a', href=True)
    try:
      href = a['href']
      if not href:
        i += 1
        fetch_apple_mobility_report_url(i)
      else:  
        urls['apple_vehicle_mobility_report'] = href
        browser.close()
    except:
      i += 1
      fetch_apple_mobility_report_url(i)

def fetch_datasets(temp_dir):
  try:
    os.mkdir(temp_dir)
  except:
    print("temp directory already created")
  finally:
    for k in urls:
      if k == 'apple_covid_dashboard':
        continue
      req = requests.get(urls[k])
      if req.ok:
        p = re.compile(r'.*(\.[a-z]+)$')
        m = p.match(urls[k])
        file_ext = m.group(1)
        filename = temp_dir + 'data_' + k + file_ext
        with open(filename, 'wb') as f:
          f.write(req.content)

def prep_datasets(temp_dir):
  # handle selecting google dataset (from zip)
  path_zip = temp_dir + 'data_' + 'google_regional_mobility_report.zip'
  path_out = temp_dir + 'google_regional_mobility_report'
  with ZipFile(path_zip, 'r') as zip_buffer:
    zip_buffer.extractall(path_out)

def update_dataset(dir):
  if type(dir) == str:
    fetch_apple_mobility_report_url()
    fetch_datasets(dir)
    prep_datasets(dir)
  else:
    print("arg must be a string")

def print_separator(ch='=', l=48):
  s = ch * l
  print(s)

def calc_n_week_average(index, row, weeks, df):
  n_sum = 0
  selection_df = df[index - 7 * weeks:index]
  for i in selection_df:
    n_sum += i
  print(f'avg: {float(n_sum / 7 * weeks)}')
  return float(n_sum / 7 * weeks)

def calc_n_week_std(index, weeks, df):
  if(index - 1 < 7 * weeks):
    return 0
  else:
    return np.std(df[index - 7 * weeks:index])

def assign_school_type(string):
  school = string.lower()
  if 'elementary' in school:
    return 1
  elif 'intermediate' in school:
    return 2
  elif 'middle' in school:
    return 3
  elif 'high school' in school:
    return 4
  elif 'college' or 'academy' or 'university' or 'institute' in school:
    return 5
  else:
    return 0

def format_date(date):
  p = re.compile('([0-9]?[0-9])\/([0-9]?[0-9])\/([0-9][0-9])')
  m = p.match(date)
  month = '0' + m.group(1) if len(m.group(1)) < 2 else m.group(1)
  day =  '0' + m.group(2) if len(m.group(2)) < 2 else m.group(2)
  year = m.group(3)
  year = year + '20'
  f = year + '-' + month + '-' + day
  return f