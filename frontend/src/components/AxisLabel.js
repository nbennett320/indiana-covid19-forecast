import React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import { Typography } from '@material-ui/core'
import { Label } from 'recharts'

const AxisLabel = props => {
  const classes = useStyles()
  return (
    <Label
      viewBox={{
        ...props.viewBox
      }}
    >
      {/* <Typography
        variant='body2'
        color='textPrimary'
      > */}
        { props.value }
      {/* </Typography> */}
    </Label>
  )
}

const useStyles = makeStyles(() => ({
  xLabel: {
    height: '20px',
    width: '20px',
    position: 'relative',
    padding: '6px 0',
    top: '40px'
  },
}))

export default AxisLabel