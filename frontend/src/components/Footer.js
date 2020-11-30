import React from 'react'
import { makeStyles } from '@material-ui/core/styles'

const Footer = props => {
  const classes = useStyles()
  return (
    <div className={classes.main}>
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: '100%',
    minHeight: '64px',
    backgroundColor: 'rgba(0,0,0,0.01)',
  }
}))

export default Footer