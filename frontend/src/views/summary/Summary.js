import React from 'react'
import DataPlot from './DataPlot'

const Summary = props => {
  return (
    <div style={styles}>
      <DataPlot 
        county={"Indiana"}
        plotData={"covid_count"}
        format={{
          title: "Indiana Covid-19 Forecast",
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
  backgroundColor: '#fafafa'
}

export default Summary