import React from 'react'
import Plot from 'react-plotly.js'
import porter_county from '../../data/model_prediction_Porter_covid_count.json'

const Plots = props => {
  console.log(porter_county)
  return (
    <div style={styles}>
      <Plot
        data={[
          {
            x: porter_county.x_pred,
            y: porter_county.y_pred,
            name: 'Forecasted',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#ffc107'
            }
          },
          {
            x: porter_county.x_data,
            y: porter_county.y_data,
            name: 'Cases',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#00bcd4'
            }
          },
          {
            x: [porter_county.x_data[porter_county.x_data.length - 1], porter_county.x_pred[0]],
            y: [porter_county.y_data[porter_county.y_data.length - 1], porter_county.y_pred[0]],
            name: 'Today',
            type: 'line',
            mode: 'lines',
            marker: {
              color: '#0097a7'
            }
          }
        ]}
        layout={{
          title: `${porter_county.county} County COVID-19 forecast.`,
          xaxis: {
            title: 'Date',
          },
          yaxis: {
            title: 'Infections per day',
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

export default Plots