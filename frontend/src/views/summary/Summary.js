import React from 'react'
import { default as Plot } from './PlotContainer'

const Summary = props => {
  return (
    <div style={styles}>
      <Plot
        county="Porter"
        plotData="covid_count"
        format={{
          title: "Indiana County Covid-19 Forecast",
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
  backgroundColor: '#fafafa',
  paddingTop: 'calc(32px + 4px)'
}

export default Summary