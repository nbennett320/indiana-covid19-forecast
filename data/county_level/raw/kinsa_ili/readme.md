# Kinsa Influenza-like Illness Weather Map

- **Data source**: https://www.kinsahealth.co/

- **Last downloaded**: updated daily

- **Data description**: contains measures of anomalous influenza-like illness incidence (ILI) outbreaks in real-time using Kinsa’s county-level illness signals, developed from real-time geospatial thermometer data, and highly accurate 12-week illness forecasts

- **Known data quality issues**: data from Alaska and Hawaii are not available

- **Short list of data columns**: 
	- **region_id**: county FIPS
	- **region_name**: full name of county including "County"
	- **state**: state abbreviation
	- **date**: date of data in YYYY-MM-DD format
	- **observed_ili**: daily ILI incidence in the specified region on the specified date; null starting on current date
	- **atypical_ili**: will contain the observed ILI if it is atypical; otherwise is null
	- **anomaly_diff**: measure of how much atypical illness is present, quantified by subtracting the upper bound of the typical illness range from the observed illness; equals 0 if the observed illness is in the typical range or lower than typical. Null starting on current date, otherwise if anomaly_diff is null Kinsa does not have enough data from the county to identify atypical illness level
	- **forecast_expected**: where illness is expected to be based on time of year in given county; null before March 2
	- **forecast_lower**: lower bound for expected forecast
	- **forecast_upper**: upper bound for expected forecast

- **Notes**: 
	- data begins Feb 16 and ends May 24
	- See FAQs here: https://content.kinsahealth.com/en-us/atypical-illness-faq
	- See technical details on methodology here: https://content.kinsahealth.com/covid-detection-technical-approach
	- Thank you to Kinsa Inc for sharing their data with us.


