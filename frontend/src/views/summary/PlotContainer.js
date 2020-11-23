import React, { useState } from 'react'
import DataPlot from './DataPlot'
import PlotOptions from './PlotOptions'
import { makeStyles } from '@material-ui/core/styles'

const PlotContainer = props => {
  const [showData, setShowData] = useState(true)
  const [showSmooth, setShowSmooth] = useState(false)
  const [smoothType, setSmoothType] = useState('polynomial')
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <div className={classes.rowContainer}>
        {/* <PlotOptions 
          showData={showData}
          showSmooth={showSmooth}
          smoothType={smoothType}
          toggleShowData={() => setShowData(!showData)}
          toggleShowSmooth={() => setShowSmooth(!showSmooth)}
          changeSmoothType={val => setSmoothType(val)}
        /> */}
        <DataPlot 
          {...props}
          showData={showData}
          showSmooth={showSmooth}
          smoothType={smoothType}
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