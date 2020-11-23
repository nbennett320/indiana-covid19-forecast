import React from 'react'
import { default as Plot } from './PlotContainer'

const Summary = props => {
  return (
    <div style={styles}>
      <Plot
        county={props.county}
        plotData="covid_count"
        format={{
          title: `${props.county} County Covid-19 Forecast`,
          xLab: "Date",
          yLab: "Cases per day"
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