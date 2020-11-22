import React from 'react'
import Plot from 'react-plotly.js'
import moment from 'moment'

const getRenderedData = props => {
  const rawData = require(`../../data/model_prediction_${props.county}_${props.plotData}.json`)
  const {
    x_data,
    y_data,
    x_pred,
    y_pred,
    x_data_polynomial,
    y_data_polynomial,
    x_pred_polynomial,
    y_pred_polynomial,
  } = rawData
  const data = [
    {
      x: x_data.map(t => new Date(t / Math.pow(10, 6))),
      y: y_data,
      name: 'Cases',
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#00bcd4'
      }
    },
    {
      x: x_pred.map(t => new Date(t / Math.pow(10, 6))),
      y: y_pred,
      name: 'Forecasted',
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#ffc107'
      }
    },
    {
      x: [x_data[x_data.length - 1], x_pred[0]].map(t => new Date(t / Math.pow(10, 6))),
      y: [y_data[y_data.length - 1], y_pred[0]],
      name: 'Today',
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#0097a7'
      }
    }
  ]
  if(props.showSmooth) {
    if(props.smoothType === 'polynomial') {
      data.push(...[
        {
          x: x_data_polynomial.map(t => new Date(t / Math.pow(10, 6))),
          y: y_data_polynomial,
          name: 'Cases (Smooth)',
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#d43c00'
          }
        },
        {
          x: x_pred_polynomial.map(t => new Date(t / Math.pow(10, 6))),
          y: y_pred_polynomial,
          name: 'Forecasted (Smooth)',
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#8307ff'
          }
        },
        {
          x: [x_data_polynomial[x_data_polynomial.length - 1], x_pred_polynomial[0]].map(t => new Date(t / Math.pow(10, 6))),
          y: [y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]],
          name: 'Today (Smooth)',
          showLegend: false,
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#a72d00'
          }
        }
      ])
    }
  }
  return data
}

const DataPlot = props => {
  const data = getRenderedData({...props})
  console.log(data)
  return (
    <div style={styles}>
      <Plot
        data={data}
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