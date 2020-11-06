import pandas as pd
import numpy as np
import requests, lxml, bs4
import os, sys, urllib, re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url_jh_cases = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
url_jh_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
url_google_regional_mobility_report = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
url_apple_covid_dashboard = "https://covid19.apple.com/mobility"
url_apple_vehicle_mobility_report = ""
url_indiana_hospital_vent_data = "https://hub.mph.in.gov/dataset/4d31808a-85da-4a48-9a76-a273e0beadb3/resource/0c00f7b6-05b0-4ebe-8722-ccf33e1a314f/download/covid_report_bedvent_date.xlsx"

# initialize data
df_jh_cases = None
df_jh_deaths = None
df_google_regional_mobility_report = None
df_apple_covid_dashboard = None
df_apple_vehicle_mobility_report = None
df_indiana_hospital_vent_data = None

# temp directory name (for csv datas)
temp_dir = './._temp_'

def fetch_apple_mobility_report_url(i=0):
  browser = webdriver.Chrome('/snap/bin/chromium.chromedriver')
  browser.get(url=url_apple_covid_dashboard)
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
      global url_apple_vehicle_mobility_report
      url_apple_vehicle_mobility_report = href
      browser.close()
  except:
    i += 1
    fetch_apple_mobility_report_url(i)

def fetch_datasets():
  try:
    os.mkdir(temp_dir)
  except:
    print("temp directory already created")
  finally:
    urls = [
      url_jh_cases,
      url_jh_deaths,
      url_google_regional_mobility_report,
      url_apple_vehicle_mobility_report,
      url_indiana_hospital_vent_data,
    ]
    for url in urls:
      file = requests.get(url)
      data_buffer = file.content.decode('utf-8')
      data = pd.read_csv(data_buffer)
      print('data', data)

def main():
  fetch_apple_mobility_report_url()
  fetch_datasets()

def test():
  fetch_apple_mobility_report_url()
  print(url_apple_vehicle_mobility_report)
  fetch_datasets()

# main()
test()