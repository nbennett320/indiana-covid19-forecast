import React from 'react'
import DataPlot from './DataPlot'
import { makeStyles } from '@material-ui/core/styles'
import { Typography } from '@material-ui/core'

const PlotContainer = props => {
  const classes = useStyles()
  const domainLength = () => {
    const { viewRange } = props
    switch(viewRange) {
      case 'month':
        return 31
      default:
        // all shown
        return -1
    }
  }
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
          domainLength={domainLength()}
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