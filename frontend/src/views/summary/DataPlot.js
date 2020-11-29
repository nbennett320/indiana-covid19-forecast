import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Label,
  ReferenceLine
} from 'recharts'
import { makeStyles } from '@material-ui/core/styles'

const formatDate = date => {
  const reg = date.matchAll(/([0-9]?[0-9])-([0-9]?[0-9])-([0-9][0-9][0-9][0-9])/gm)
  const match = [...reg][0]
  const month = match[1].length > 1
    ? match[1]
    : `0${match[1]}`
  const day = match[2].length > 1
    ? match[2]
    : `0${match[2]}`
  return `${month}-${day}-${match[3]}`
}

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
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        [`y_${props.format.dataLab}_data`]: y_data[i],
        name: props.format.dataLab
      })),
      ...[x_data[x_data.length - 1], x_pred[0]].map((t, i) => ({
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        [`y_${props.format.dataLab}_today`]: Math.round([y_data[y_data.length - 1], y_pred[0]][i]),
        name: 'Today'
      })),
      ...x_pred.map((t, i) => ({
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        [`y_${props.format.dataLab}_forecasted`]: Math.round(y_pred[i]),
        name: 'Forecasted'
      }))
    ])
  }
  if(props.showSmooth) {
    if(props.smoothingMethod === 'polynomial') {
      data.push(...[
        ...x_data_polynomial.map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          [`y_${props.format.dataLab}_data`]: y_data_polynomial[i],
          name: props.format.dataLab
        })),
        ...[x_data_polynomial[x_data_polynomial.length - 1], x_pred_polynomial[0]].map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          [`y_${props.format.dataLab}_today`]: [y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]][i],
          name: 'Today'
        })),
        ...x_pred_polynomial.map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          [`y_${props.format.dataLab}_forecasted`]: y_pred_polynomial[i],
          name: 'Forecasted'
        }))
      ])
    }
  }
  return data.flat()
}

const DataPlot = props => {
  let data = getRenderedData({...props})
  console.log('data1',data)
  const { 
    xLab,
    yLab,
    dataLab, 
    animationOffset,
    animationDuration 
  } = props.format
  const classes = useStyles()
  if(props.viewRange === 'month') {
    const d = formatDate(
      new Date(
        new Date() - 1000 * 60 * 60 * 24 * props.domainLength
      ).toLocaleDateString().replaceAll('/','-')
    )
    data = data.filter(el => {
      const elArr = el.x.split('-').map(s => Number(s))
      const dArr = d.split('-').map(s => Number(s))
      if(elArr[2] < dArr[2])
        return false
      else if(elArr[0] < dArr[0])
        return false
      else if((elArr[0] === dArr[0]) && (elArr[1] < dArr[1]))
        return false
      else return true
    })
  }
  console.log("data",data)
  return (
    <div className={classes.main}>
      <ResponsiveContainer
       height={props.userDevice.vpHeight * 0.7}
       width={props.userDevice.vpWidth * 0.9}
      >
        <LineChart 
          data={data}
          padding={{ 
            top: 12, 
            right: 48, 
            left: 48, 
            bottom: 12 
          }}
          margin={{ 
            top: 12, 
            right: 40, 
            left: 40, 
            bottom: 12 
          }}
        >
          <CartesianGrid strokeDasharray='3 3' />
          <XAxis 
            dataKey='x'
            name={xLab}
            domain={['dataMin', 'dataMax']}
            interval='preserveStartEnd'
            allowDuplicatedCategory={true}
            // xAxisId='xAxisTimeSeries'
            padding={{
              top: 20,
              bottom: 20,
              left: 0,
              right: 0
            }}
            label={labelProps => {
              return (
                <Label
                  viewBox={{
                    ...labelProps.viewBox,
                    y: labelProps.viewBox.y + 20,
                  }}
                >
                  {xLab}
                </Label>
              )
            }}
          />
          <YAxis 
            label={{ 
              value: yLab,
              angle: -90,
              content: labelProps => {
                return (
                  <Label
                    angle={labelProps.angle}
                    margin={{
                      bottom: 20
                    }}
                    viewBox={{
                      ...labelProps.viewBox,
                      x: labelProps.viewBox.x - 27.5,
                    }}
                  >
                    {labelProps.value}
                  </Label>
                )
              }
            }}
          />
          <Legend 
            verticalAlign='top'
            align='right'
            margin={{
              bottom: 10
            }}
          />
          <Tooltip />
          <Line 
            type='monotone'
            dataKey={`y_${dataLab}_data`}
            // xAxisId='xAxisTimeSeries'
            stroke='#00bcd4'
            activeDot={{ r: 4 }}
            dot={false}
            legendType='line'
            formatter={n => Math.round(n)}
            name={dataLab}
            animationBegin={0 + animationOffset}
            animationDuration={animationDuration}
          />
          {/* <Line 
            type='monotone'
            dataKey={`y_${dataLab}_today`}
            // xAxisId='xAxisTimeSeries'
            stroke='#00a5bb'
            activeDot={{ r: 4 }}
            dot={false}
            legendType='none'
            formatter={n => Math.round(n)}
            animationBegin={animationDuration + animationOffset}
            animationDuration={10}
          /> */}
          <Line 
            type='monotone'
            dataKey={`y_${dataLab}_forecasted`}
            // xAxisId='xAxisTimeSeries'
            stroke='#ffc107'
            activeDot={{ r: 4 }}
            dot={false}
            legendType='line'
            formatter={n => Math.round(n)}
            name={'Forecasted'}
            animationBegin={animationDuration + animationOffset + 10}
            animationDuration={animationDuration}
          />
          <ReferenceLine 
            x={data.length - 15}
            label={labelProps => {
              return (
                <Label
                  viewBox={{
                    ...labelProps.viewBox,
                    x: labelProps.viewBox.x,
                    y: labelProps.viewBox.y + props.userDevice.vpHeight * 0.7 * 0.42
                  }}
                >
                  Today
                </Label>
              )
            }}
            stroke='#00d482'
            strokeDasharray="5 5"
            strokeWidth={1.2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: 'auto',
    margin: '0 auto',
  },
  xLabel: {
    height: '20px',
    position: 'relative',
    padding: '6px 0',
    top: '40px'
  },
}))

export default DataPlot