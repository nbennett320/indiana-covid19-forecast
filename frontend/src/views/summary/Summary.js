import React from 'react'
import { default as Plot } from './PlotContainer'
import { makeStyles } from '@material-ui/core/styles'

const Summary = props => {
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <Plot
        {...props}
        plotData="covid_count"
        isCountyLevelData={true}
        format={{
          title: `${props.county} ${props.county === 'Indiana' ? '' : 'County '}Covid-19 cases per day`,
          xLab: "Date",
          yLab: "Cases per day",
          dataLab: "Cases",
        }}
      />
      <Plot
        {...props}
        plotData="covid_deaths"
        isCountyLevelData={true}
        format={{
          title: `${props.county} ${props.county === 'Indiana' ? '' : 'County '}Covid-19 deaths per day`,
          xLab: "Date",
          yLab: "Deaths per day",
          dataLab: "Deaths",
        }}
      />
      <Plot
        {...props}
        plotData="hospital_occupation"
        isCountyLevelData={false}
        format={{
          title: `Indiana ICU bed availability`,
          xLab: "Date",
          yLab: "Available ICU Beds",
          dataLab: "Beds",
        }}
      />
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: '100%',
    height: '100%',
    backgroundColor: '#fcfcfc',
    paddingTop: '64px',
    paddingBottom: '64px'
  }
}))

export default Summary