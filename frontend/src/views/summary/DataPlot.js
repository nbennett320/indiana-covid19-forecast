import React from 'react'
import Plot from 'react-plotly.js'
import t from '../../data/model_prediction_Indiana_covid_count.json'

console.log(t)
const DataPlot = async props => {
  const path = `../../data/model_prediction_${props.county}_${props.plotData}.json`
  const {
    x_data,
    y_data,
    x_pred,
    y_pred
  } = await import(path).then(module => module)

  return (
    <div style={styles}>
      <Plot
        data={[
          {
            x: x_pred,
            y: y_pred,
            name: 'Forecasted',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#ffc107'
            }
          },
          {
            x: x_data,
            y: y_data,
            name: 'Cases',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#00bcd4'
            }
          },
          {
            x: [x_data[x_data.length - 1], x_pred[0]],
            y: [y_data[y_data.length - 1], y_pred[0]],
            name: 'Today',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#0097a7'
            }
          }
        ]}
        layout={{
          title: props.format.title,
          xaxis: {
            title: props.format.xLab,
          },
          yaxis: {
            title: props.format.yLab,
          },
          paper_bgcolor: '#fafafa',
          plot_bgcolor: '#fafafa',
        }}
      />
    </div>
  )
}

const styles = {
  width: '100%',
  paddingTop: '20px',
  margin: '0 auto',
}

export default DataPlot