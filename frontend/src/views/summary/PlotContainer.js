import React from 'react'
import DataPlot from './DataPlot'
import { makeStyles } from '@material-ui/core/styles'
import { Typography } from '@material-ui/core'
import { 
  getRenderedData,
  formatData
} from '../../util'

const getDomainLength = (viewRange) => {
  switch(viewRange) {
    case 'month':
      return 31
    case '3month':
      return 92
    default:
      return -1
  }
}

const PlotContainer = props => {
  const classes = useStyles()
  const data = formatData(
    getRenderedData({...props}),
    props.viewRange,
    getDomainLength(props.viewRange)
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