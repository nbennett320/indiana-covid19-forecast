import React from 'react'
import Plot from 'react-plotly.js'

const getRenderedData = props => {
  const rawData = props.isCountyLevelData 
    ? require(`../../data/model_prediction_${props.county}_${props.plotData}.json`)
    : require(`../../data/model_prediction_${props.plotData}.json`)
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
      name: props.format.dataLab,
      showlegend: true,
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#00bcd4'
      },
      layout: {
        showLegend: true
      }
    },
    {
      x: x_pred.map(t => new Date(t / Math.pow(10, 6))),
      y: y_pred.map(n => Math.round(n)),
      name: 'Forecasted',
      showlegend: true,
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#ffc107'
      },
    },
    {
      x: [x_data[x_data.length - 1], x_pred[0]].map(t => new Date(t / Math.pow(10, 6))),
      y: [y_data[y_data.length - 1], y_pred[0]].map(n => Math.round(n)),
      name: 'Today',
      showlegend: false,
      hoverinfo: 'skip',
      type: 'line',
      mode: 'lines',
      marker: {
        color: '#0097a7'
      },
    }
  ]
  if(props.showSmooth) {
    if(props.smoothType === 'polynomial') {
      data.push(...[
        {
          x: x_data_polynomial.map(t => new Date(t / Math.pow(10, 6))),
          y: y_data_polynomial,
          name: 'Cases (Smooth)',
          showlegend: true,
          hoverinfo: 'skip',
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#d43c00',
          },
        },
        {
          x: x_pred_polynomial.map(t => new Date(t / Math.pow(10, 6))),
          y: y_pred_polynomial,
          name: 'Forecasted (Smooth)',
          showlegend: true,
          hoverinfo: 'skip',
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#8307ff'
          },
          layout: {
          }
        },
        {
          x: [x_data_polynomial[x_data_polynomial.length - 1], x_pred_polynomial[0]].map(t => new Date(t / Math.pow(10, 6))),
          y: [y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]],
          name: 'Today (Smooth)',
          showlegend: false,
          hoverinfo: 'skip',
          type: 'line',
          mode: 'lines',
          marker: {
            color: '#a72d00'
          },
        }
      ])
    }
  }
  return data
}

const DataPlot = props => {
  const data = getRenderedData({...props})
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
          autosize: true,
          width: 1000,
          height: 700,
          paper_bgcolor: '#fcfcfc',
          plot_bgcolor: '#fcfcfc',
        }}
      />
    </div>
  )
}

const styles = {
  width: 'auto',
  margin: '0 auto',
}

export default DataPlot