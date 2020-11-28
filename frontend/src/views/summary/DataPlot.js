import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

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
  x_data.map((t, i) => ({
    x: new Date(t / Math.pow(10, 6)),
    y: y_data[i],
    name: props.format.dataLab
  }))
  const data = []
  if(!props.showSmooth) {
    data.push(...[
      ...x_data.map((t, i) => ({
        x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
        [`y_${props.format.dataLab}_data`]: y_data[i],
        name: props.format.dataLab
      })),
      ...x_pred.map((t, i) => ({
        x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
        [`y_${props.format.dataLab}_forecasted`]: Math.round(y_pred[i]),
        name: 'Forecasted'
      })),
      ...[x_data[x_data.length - 1], x_pred[0]].map((t, i) => ({
        x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
        [`y_${props.format.dataLab}_today`]: Math.round([y_data[y_data.length - 1], y_pred[0]][i]),
        name: 'Today'
      }))
    ])
  }
  if(props.showSmooth) {
    if(props.smoothingMethod === 'polynomial') {
      data.push(...[
        ...x_data_polynomial.map((t, i) => ({
          x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
          [`y_${props.format.dataLab}_data`]: y_data_polynomial[i],
          name: props.format.dataLab
        })),
        ...x_pred_polynomial.map((t, i) => ({
          x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
          [`y_${props.format.dataLab}_forecasted`]: y_pred_polynomial[i],
          name: 'Forecasted'
        })),
        ...[x_data_polynomial[x_data_polynomial.length - 1], x_pred_polynomial[0]].map((t, i) => ({
          x: new Date(t / Math.pow(10, 6)).toLocaleDateString(),
          [`y_${props.format.dataLab}_today`]: [y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]][i],
          name: 'Today'
        }))
      ])
    }
  }
  return data.flat()
}

const formatTicks = tickItem => {
  console.log(tickItem)
  return tickItem.toLocaleDateString()
}

const DataPlot = props => {
  const data = getRenderedData({...props})
  const { dataLab } = props.format
  console.log("data",data)
  return (
    <div style={styles}>
      <ResponsiveContainer
        height={700}
        width={1000}
      >
        <LineChart data={data}>
          <CartesianGrid strokeDasharray='3 3' />
          <XAxis dataKey='x' />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line 
            type='monotone'
            dataKey={`y_${dataLab}_data`}
            stroke='#00bcd4'
            activeDot={{ r: 4 }}
            dot={false}
            name={dataLab}
          />
          <Line 
            type='monotone'
            dataKey={`y_${dataLab}_forecasted`}
            stroke='#ffc107'
            activeDot={{ r: 4 }}
            dot={false}
            name={'Forecasted'}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const styles = {
  width: 'auto',
  margin: '0 auto',
}

export default DataPlot