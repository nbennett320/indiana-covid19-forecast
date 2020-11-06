mobility_data4 <- read.csv('./data/2020_US_Region_Mobility_Report_indiana_county_level.csv')
mobility_df4 <- data.frame(mobility_data4)

main <- function() {
  print(sapply(mobility_df4, typeof))
  for(i in 1:length(mobility_df4$grociery_pharmacy)) {
    mobility_df4$grociery_pharmacy[i] <- mobility_df4$grociery_pharmacy[i] + 100
  }

  for(i in 1:length(mobility_df4$parks)) {
    mobility_df4$parks[i] <- mobility_df4$parks[i] + 100
  }

  for(i in 1:length(mobility_df4$residential)) {
    mobility_df4$residential[i] <- mobility_df4$residential[i] + 100
  }

  for(i in 1:length(mobility_df4$retail_recreation)) {
    mobility_df4$retail_recreation[i] <- mobility_df4$retail_recreation[i] + 100
  }

  for(i in 1:length(mobility_df4$transit_stations)) {
    mobility_df4$transit_stations[i] <- mobility_df4$transit_stations[i] + 100
  }

  for(i in 1:length(mobility_df4$workplace)) {
    mobility_df4$workplace[i] <- mobility_df4$workplace[i] + 100
  }

  print(mobility_df4)
  write.csv(mobility_df4, "./data/mobility_adjusted_df3.csv")
}

main()