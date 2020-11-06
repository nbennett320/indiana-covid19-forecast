## how to update data set
### df format method
  1. fetch john hobkins time series data (cases, deaths) from https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv and https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv
  2. format jh time series data
    - format time series dates to yyyy-mm-dd
      - `scripts/format_time_series_dates.py`
    - filter data relevent to indiana and append census data
      - `scripts/format_time_series.py`
  3. fetch google regional mobility report from https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
  4. unzip and grab `2020_US_Region_Mobility_Report.csv`
  5. format google mobility report
    - normalize reported mobility
      - `scripts/normalize_time_series_mobility_2.py`
    - sort by date (instead of by county)
      - `scripts/sort_frames.py`
    - filter and append relevant data
      - `scripts/time_series_mobility.py`
  6. fetch apple vehicle mobility data from https://covid19-static.cdn-apple.com/covid19-mobility-data/2019HotfixDev30/v3/en-us/applemobilitytrends-2020-11-04.csv
  7. format apple mobility data
    - filter relevant data and append
      - `scripts/time_series_vehicle.py`
  8. fetch indiana hospital and vent data from https://hub.mph.in.gov/dataset/4d31808a-85da-4a48-9a76-a273e0beadb3/resource/0c00f7b6-05b0-4ebe-8722-ccf33e1a314f/download/covid_report_bedvent_date.xlsx
  9. convert to csv and append hospital and vent data
    - convert xlsx data to csv
      - `scripts/convert_xlsx_to_csv.py`
    - append
      - `scripts/append_hospital_series.py`
    