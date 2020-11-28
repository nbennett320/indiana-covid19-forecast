import React, { useState } from 'react'
import DataPlot from './DataPlot'
import { makeStyles } from '@material-ui/core/styles'

const PlotContainer = props => {
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <div className={classes.rowContainer}>
        <DataPlot 
          {...props}
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
    paddingBttom: '12px'
  },
  rowContainer: {
    display: 'flex',
    flexDirection: 'row'
  }
}))

export default PlotContainer