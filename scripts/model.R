data1 <- read.csv('./data/indiana_county_level_time_series_data_formatted_trimmed_date.csv')
data2 <- read.csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv')
df1 <- data.frame(data1)
df2 <- data.frame(data2)

main <- function() {
  # model <- df1$date ~ avg_cases_by_week + grociery_pharmacy_mobility + parks_mobility + retail_recreation_mobility + transit_stations_mobility + workplace_mobility + driving_mobility_vehicle + transit_mobility_vehicle + walking_mobility_instance + std_last_week_grociery_pharmacy_mobility + std_last_week_park_mobility + std_last_week_retail_recreation_mobility + std_last_week_transit_station_mobility + std_last_week_workplace_mobility

  model <- df1$date ~ cases + deaths + lat + long + avg_cases_last_week + avg_deaths_last_week + avg_cases_last_2_weeks + avg_deaths_last_2_week + std_cases_last_week + std_deaths_last_week + std_cases_last_2_weeks + std_deaths_last_2_weeks + median_gross_rent + average_household_size + building_permits_number + percent_households_with_computer + percent_households_with_broadband_internet + dollars_per_capita_income_in_past_12_months_2018 + population_per_square_mile + median_household_income + percent_housing_units_in_multi_unit_structures + percent_under_65_without_health_insurance + percent_under_65_with_disability
  f <- glm(model, data=data1, family=gaussian)
  sink(file='./out.txt')
  summary(f)
}

main()