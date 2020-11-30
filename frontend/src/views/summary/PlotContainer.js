import React from 'react'
import DataPlot from './DataPlot'
import { makeStyles } from '@material-ui/core/styles'
import { Typography } from '@material-ui/core'
import { 
  getRenderedData,
  formatData
} from '../../util'

const animationDuration = 1000
const animationOffset = 0

const getDomainLength = (viewRange) => {
  switch(viewRange) {
    case 'month':
      return 31
    case '3month':
      return 92
    default:
      const startDay = new Date(2020, 2, 25)
      const today = new Date()
      return Math.round(Math.abs((today - startDay) / (24 * 60 * 60 * 1000)))
  }
}

const PlotContainer = props => {
  const classes = useStyles()
  const domainLength = getDomainLength(props.viewRange)
  const data = formatData(
    getRenderedData({...props}),
    props.viewRange,
    domainLength
  )
  return (
    <div className={classes.main}>
      <div className={classes.rowContainer}>
        <Typography 
          className={classes.title}
          variant='h6'
          color='textPrimary'
        >
          { props.format.title }
        </Typography>
        
        <DataPlot 
          {...props}
          data={data}
          predictionLength={14}
          domainLength={domainLength}
          animationDuration={animationDuration}
          animationOffset={animationOffset}
        />
      </div>
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: 'auto',
    height: 'auto',
    backgroundColor: '#fcfcfc',
    display: 'flex',
    flexDirection: 'column',
    paddingTop: '12px',
    paddingBttom: '24px'
  },
  rowContainer: {
    display: 'flex',
    flexDirection: 'column'
  },
  title: {
    marginLeft: 'auto',
    marginRight: 'auto'
  }
}))

export default PlotContainer