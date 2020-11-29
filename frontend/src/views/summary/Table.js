import React from 'react'
import {
  DataTable
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'

const Table = props => {
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <DataTable />
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {

  }
}))

export default Table