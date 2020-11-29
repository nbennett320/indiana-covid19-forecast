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

const DataPlot = props => {
  const { 
    xLab,
    yLab,
    dataLab, 
    animationOffset,
    animationDuration 
  } = props.format
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <ResponsiveContainer
        height={props.userDevice.vpHeight * 0.7}
        width={props.userDevice.vpWidth * 0.9}
      >
        <LineChart 
          data={props.data}
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
            x={props.data.length - props.predictionLength - 1}
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