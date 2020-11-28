import React from 'react'
import { default as Plot } from './PlotContainer'

const animationDuration = 500
const Summary = props => {
  return (
    <div style={styles}>
      <Plot
        {...props}
        plotData="covid_count"
        isCountyLevelData={true}
        format={{
          title: `${props.county} ${props.county === 'Indiana' ? '' : 'County '}Covid-19 cases per day`,
          xLab: "Date",
          yLab: "Cases per day",
          dataLab: "Cases",
          animationOffset: 0,
          animationDuration: animationDuration
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
          animationOffset: animationDuration,
          animationDuration: animationDuration
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
          animationOffset: 2 * animationDuration,
          animationDuration: animationDuration
        }}
      />
    </div>
  )
}

const styles = {
  width: '100%',
  height: '100%',
  backgroundColor: '#fcfcfc',
  paddingTop: '64px',
}

export default Summary