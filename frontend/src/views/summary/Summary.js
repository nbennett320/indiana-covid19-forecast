import React from 'react'
import { default as Plot } from './PlotContainer'

const Summary = props => {
  return (
    <div style={styles}>
      <Plot
        county={props.county}
        plotData="covid_count"
        isCountyLevelData={true}
        format={{
          title: `${props.county} ${props.county === 'Indiana' ? '' : 'County'} Covid-19 cases per day`,
          xLab: "Date",
          yLab: "Cases per day",
          dataLab: "Cases"
        }}
      />
      <Plot
        county={props.county}
        plotData="hospital_occupation"
        isCountyLevelData={false}
        format={{
          title: `Indiana ICU bed availability`,
          xLab: "Date",
          yLab: "Available ICU Beds",
          dataLab: "Beds"
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