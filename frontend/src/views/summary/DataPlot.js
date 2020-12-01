import React from 'react'
import {
  ComposedChart,
  Line,
  Area,
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
import { palette } from '../../util'

const DataPlot = props => {
  const { 
    xLab,
    yLab,
    animationOffset,
    animationDuration 
  } = props.format
  const { data } = props.data
  const classes = useStyles()
  const percentage = Math.floor((props.predictionLength) / (props.domainLength + props.predictionLength) * 100)
  const percentageLower = Math.floor((props.predictionLength + 1) / (props.domainLength + props.predictionLength) * 100)
  return (
    <div className={classes.main}>
      <ResponsiveContainer
        height={props.userDevice.vpHeight * 0.7}
        width={props.userDevice.vpWidth * 0.9}
      >
        <ComposedChart 
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
          <defs>
            <linearGradient id="line-segment" x1="0" y1="0" x2="100%" y2="0">
              <stop offset="0%" stopColor={`${palette.fill.data}`} />
              <stop offset={`${100 - percentageLower}%`} stopColor={`${palette.fill.data}`} />
              <stop offset={`${100 - percentage}%`} stopColor={`${palette.fill.prediction}`} />
              <stop offset="100%" stopColor={`${palette.fill.prediction}`} />
            </linearGradient>
            <linearGradient id="area-segment" x1="0" y1="0" x2="100%" y2="0">
              <stop offset="0%" stopColor={`${palette.fill.hidden}`} />
              <stop offset={`${100 - percentageLower}%`} stopColor={`${palette.fill.hidden}`} />
              <stop offset={`${100 - percentage}%`} stopColor={`${palette.fill.predictionArea}`} />
              <stop offset="100%" stopColor={`${palette.fill.predictionArea}`} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray='3 3' />
          <XAxis 
            dataKey='x'
            name={xLab}
            allowDuplicatedCategory={true}
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
                    fill={`${palette.fill.text}`}
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
              top: 0,
              left: 'auto',
              right: 0,
              bottom: 10
            }}
            fill={`${palette.fill.text}`}
          />
          <Tooltip fill={`${palette.fill.text}`} />
          <Line 
            type='monotone'
            dataKey={`y_all`}
            stroke='url(#line-segment)'
            activeDot={{ r: 4 }}
            dot={false}
            legendType='line'
            formatter={n => Math.round(n)}
            name={'Data (blue), Forecasted (yellow)'}
            animationBegin={animationOffset}
            animationDuration={animationDuration}
          />
          <Area
            type='monotone'
            dataKey={`y_pred`}
            stroke='url(#area-segment)'
            fill={`${palette.fill.predictionArea}`}
            activeDot={{ r: 4 }}
            dot={false}
            legendType='line'
            formatter={n => n}
            name={'Data (blue), Forecasted (yellow)'}
            animationBegin={animationOffset}
            animationDuration={animationDuration}
          />
          <ReferenceLine 
            x={data.length - props.predictionLength - 1}
            label={labelProps => {
              return (
                <Label
                  viewBox={{
                    ...labelProps.viewBox,
                    x: labelProps.viewBox.x,
                    y: labelProps.viewBox.y + props.userDevice.vpHeight * 0.7 * 0.42
                  }}
                  fill={`${palette.fill.textSecondary}`}
                >
                  Today
                </Label>
              )
            }}
            stroke={`${palette.fill.referenceLine}`}
            strokeDasharray="5 5"
            strokeWidth={1.2}
          />
        </ComposedChart>
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