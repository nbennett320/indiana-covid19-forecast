import React from 'react'
import DataPlot from './DataPlot'
import DataTable from './DataTable'
import { makeStyles } from '@material-ui/core/styles'
import { 
  Typography,
  Paper
} from '@material-ui/core'
import { 
  getRenderedData,
  formatData
} from '../../util'

const predictionLength = 14
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
    <Paper 
      elevation={3}
      className={classes.main}
    >
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
        predictionLength={predictionLength}
        domainLength={domainLength}
        animationDuration={animationDuration}
        animationOffset={animationOffset}
      />
      <DataTable 
        {...props}
        data={data}
        predictionLength={predictionLength}
      />
    </Paper>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: 'auto',
    height: 'auto',
    // backgroundColor: '#fcfcfc',
    display: 'flex',
    flexDirection: 'column',
    padding: '48px 0',
    margin: '24px 3%'
  },
  title: {
    marginLeft: 'auto',
    marginRight: 'auto'
  }
}))

export default PlotContainer