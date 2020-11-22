import React from 'react'
import {
  FormGroup,
  FormControl,
  FormControlLabel,
  InputLabel,
  Checkbox,
  Select,
  MenuItem
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'

const PlotOptions = props => {
  const classes = useStyles()
  return (
    <div className={classes.root}>
      <FormControl className={classes.formControl}>
        <FormGroup>
          <FormControlLabel 
            control={
              <Checkbox
                checked={props.showData}
                onChange={props.toggleShowData}
                color="primary"
              />
            }
            label="Plot data"
          />
          <FormControlLabel 
            control={
              <Checkbox
                checked={props.showSmooth}
                onChange={props.toggleShowSmooth}
                color="primary"
              />
            }
            label="Plot smooth"
          />
        </FormGroup>
        
        { props.showSmooth && <FormControl 
          className={classes.selectControl} 
          variant="outlined"
        >
          <InputLabel id="select-smooth-type-label">
            Smoothing method
          </InputLabel>
          <Select
            value={props.smoothType}
            onChange={val => props.changeSmoothType(val)}
            label="Smoothing method"
            labelId="select-smooth-type-label"
          >
            <MenuItem value="polynomial">
              Polynomial
            </MenuItem>
          </Select>
        </FormControl> }
      </FormControl>
    </div>
  )
}

const useStyles = makeStyles(theme => ({
  root: {
    display: 'flex',
    minWidth: 160,
    marginRight: theme.spacing(3)
  },
  formControl: {
    width: 160,
    margin: theme.spacing(3),
  },
  selectControl: {
    minWidth: 180,
    marginTop: theme.spacing(1.5),
  },
}))

export default PlotOptions