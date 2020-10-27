# SafeGraph Weekly Patterns Data

- **Data source**: https://www.safegraph.com/covid-19-data-consortium

- **Last downloaded**: updated weekly

- **Data description**: place foot-traffic and demographic aggregations that answer: how often people visit, where they came from, where else they go, and more; available for ~3.6MM POI in the USA; aggregated and delivered weekly (week beginning on Sunday and ending Saturday); weekly data available starting from March 1, 2020; data provided by SafeGraph

- **Known data quality issues**: 
    - Some differential privacy adjustments were made. Details provided at https://docs.safegraph.com/docs/weekly-patterns.

- **Short list of data columns**: (wide format)
	- Identifiers
	    - **countyFIPS**: string; county FIPS
	    - **n_places_{CATEGORY}**: scalar; number of places (POIs) in countyFIPS for given industry category
	    - **location_names_{CATEGORY}**: list of locations in countyFIPS for given industry category; should have length equal to n_places_{CATEGORY}
	    - **naics_codes_{CATEGORY}**: list of NAICS codes (i.e., industry categories) for each location in location_names_{CATEGORY}
	- List fields (all lists have length equal to n_places and for brevity, let loc_i be the ith POI in location_names_{CATEGORY)
	    - **raw_visitor_counts_week_{DATE}_{CATEGORY}**: list where the ith component corresponds to the number of unique visitors from panel to loc_i during the week starting on given {DATE}
	    - **raw_visit_counts_week_{DATE}_{CATEGORY}**: list where the ith component corresponds to the number of visits in panel to loc_i during the week starting on given {DATE}
	    - **visits_by_day_{DATE}_{CATEGORY}**: list where the ith component corresponds to the number of unique visitors from panel to loc_i on given date
	    - **max_visits_in_hour_{DATE}_{CATEGORY}**: list where the ith component corresponds to the max number of visits to loc_i over the span of an hour on given date
	    - **bucketed_dwell_times_{BUCKET}\_week\_{DATE}\_{CATEGORY}**: list where the ith component corresponds to the number of visits to loc_i that were within the given time {BUCKET} duration over the span of the week starting on {DATE}; {BUCKET} is a range of minutes
	- Scalar fields
	    - **median_raw_visitor_counts_week_{DATE}_{CATEGORY}**: median of raw_visitor_counts_week\_{DATE}\_{CATEGORY}
	    - **median_raw_visit_counts_week_{DATE}_{CATEGORY}**: median of raw_visit_counts_week_{DATE}\_{CATEGORY}
	    - **sum_visits_by_day_{DATE}_{CATEGORY}**: sum of visits_by_day_{DATE}\_{CATEGORY}
	    - **max_max_visits_in_hour_{DATE}_{CATEGORY}**: max of max_visits_in_hour_{DATE}\_{CATEGORY}
	    - **sum_bucketed_dwell_times_{BUCKET}\_week\_{DATE}\_{CATEGORY}**: sum of bucketed_dwell_times_{BUCKET}\_week\_{DATE}\_{CATEGORY}
	


- **Notes**:

	- There are two possible industry groupings to explore: grouping = "main" or grouping = "specialty"
		- grouping = "main" uses the two-digit NAICS codes to group industries
		- grouping = "specialty" uses a manual schema based upon NAICS codes in an attempt to create industry groups that are relevant for COVID-19 analyses; has groupings such as airports, restaurants, grocery stores, pharmacies, other retail, school, nursing homes, medsurg hospitals, entertainment, religion organizations, and more
    - This data has been cleaned already. For the original schema and details, please see https://docs.safegraph.com/docs/weekly-patterns. 
    - Note: the original data from SafeGraph is collected at the census block level, which is more granular than the county level data provided in this repo.
    - Thank you to SafeGraph for kindly sharing their data with us.