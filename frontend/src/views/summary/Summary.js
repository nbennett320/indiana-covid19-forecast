import React from 'react'
import Plots from './Plots'

const Summary = props => {
  return (
    <div style={styles}>
      Summary
      <Plots />
    </div>
  )
}

const styles = {
  width: '100%',
  height: '100%',
  backgroundColor: '#fafafa'
}

export default Summary