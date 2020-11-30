import React from 'react'
import {
  TableBody,
  TableHead,
  TableCell,
  TableContainer,
  TableRow,
  Paper
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { 
  mapViewRange,
  formatData
} from '../../util'

const TableHeader = props => {
  return (
    <TableHead>
      <TableRow>
        <TableCell>
          Max Covid-19 {props.format.yLab.toLowerCase()} ({mapViewRange(props.viewRange)})
        </TableCell>
        <TableCell>
          Min Covid-19 {props.format.yLab.toLowerCase()} ({mapViewRange(props.viewRange)})
        </TableCell>
        <TableCell>
          Max predicted Covid-19 {props.format.yLab.toLowerCase()} ({mapViewRange(props.viewRange)})
        </TableCell>
        <TableCell>
          Min predicted Covid-19 {props.format.yLab.toLowerCase()} ({mapViewRange(props.viewRange)})
        </TableCell>
      </TableRow>
    </TableHead>
  )
}

const DataTable = props => {
  const classes = useStyles()
  const timeseriesData = props.data.filter(t => {
    if(t.name === props.format.dataLab)
      return true
    else return false
  }) 
  const predictedData = props.data.filter(t => {
    if(t.name === 'Forecasted')
      return true
    else return false
  }) 
  return (
    <Paper 
      elevation={0}
      className={classes.main}
    >
      <TableContainer>
        <TableHeader 
          {...props}
        />
        <TableBody>
          <TableRow>
            <TableCell>
              {Math.max(...timeseriesData.map(e => e.y_all))}
            </TableCell>
            <TableCell>
              {Math.min(...timeseriesData.map(e => e.y_all))}
            </TableCell>
            <TableCell>
              {Math.max(...predictedData.map(e => e.y_all))}
            </TableCell>
            <TableCell>
              {Math.min(...predictedData.map(e => e.y_all))}
            </TableCell>
          </TableRow>
        </TableBody>
      </TableContainer>
    </Paper>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    margin: '48px 5%',
  }
}))

export default DataTable