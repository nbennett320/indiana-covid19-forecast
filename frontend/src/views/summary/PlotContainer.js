import React, { useState } from 'react'
import DataPlot from './DataPlot'
import PlotOptions from './PlotOptions'

const PlotContainer = props => {
  const [showData, setShowData] = useState(true)
  const [showSmooth, setShowSmooth] = useState(false)
  const [smoothType, setSmoothType] = useState('polynomial')
  return (
    <div style={styles.main}>
      <div style={styles.rowContainer}>
        <PlotOptions 
          showData={showData}
          showSmooth={showSmooth}
          smoothType={smoothType}
          toggleShowData={() => setShowData(!showData)}
          toggleShowSmooth={() => setShowSmooth(!showSmooth)}
          changeSmoothType={val => setSmoothType(val)}
        />
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

const styles = {
  main: {
    width: '100%',
    height: '100%',
    padding: '8px 12px',
    backgroundColor: '#fafafa',
    display: 'flex',
    flexDirection: 'column',
  },
  rowContainer: {
    display: 'flex',
    flexDirection: 'row'
  }
}

export default PlotContainer