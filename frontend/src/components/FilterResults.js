import React, { useState } from 'react'
import {
  Popper,
  ClickAwayListener,
  MenuList,
  MenuItem,
  Collapse,
  Paper,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import counties from '../data/indiana_counties.json'

const FilterResults = props => {
  const classes = useStyles()
  const handleSelect = (e, el) => {
    props.handleClose(e)
    props.handleChangeCounty(el)
  }

  return (
    <Popper
      open={props.isOpen}
      anchorEl={props.anchorEl}
      onClose={props.handleClose}
      anchororigin={{
        vertical: 'bottom',
        horizontal: 'left',
      }}
      transformorigin={{
        vertical: 'top',
        horizontal: 'left',
      }}
      id="filter-menu"
      className={classes.main}
      transition
      disablePortal
    >
      {({ TransitionProps }) => (
        <Collapse
          {...TransitionProps}
        >
          <Paper>
            <ClickAwayListener onClickAway={props.handleClose}>
              <MenuList
                autoFocus={false}
                id="menu-list-content"
              >
                {
                  counties['county_name']
                    .filter(el => el.toLowerCase().includes(props.query.toLowerCase()))
                    .map(el => (
                      <MenuItem 
                        onClick={e => handleSelect(e, el)}
                        key={el}
                      >
                        { el } {el.includes('Indiana') ? '(all counties)' : 'County'}
                      </MenuItem>
                    ))
                    .slice(0, 10)
                    .sort((a, b) => props.query - a < props.query - b 
                      ? props.query - a 
                      : props.query - b
                    )
                }
              </MenuList>
            </ClickAwayListener>
          </Paper>
        </Collapse>
      )}
    </Popper>
  )
}

const useStyles = makeStyles(theme =>({
  main: {
    width: '234px'
  }
}))

export default FilterResults