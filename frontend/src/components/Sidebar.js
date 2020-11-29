import React from 'react'
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Divider,
  FormControl,
  Select,
  InputLabel,
  MenuItem
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import ChevronLeft from '@material-ui/icons/ChevronLeft'
import AssessmentIcon from '@material-ui/icons/Assessment'
// import MapIcon from '@material-ui/icons/Map'
import InfoIcon from '@material-ui/icons/Info'
import TimelineIcon from '@material-ui/icons/Timeline'
// import SumIcon from '@material-ui/icons/Functions'

const Sidebar = props => {
  const classes = useStyles()
  return (
    <Drawer
      open={props.isOpen}
      onClose={props.toggleSidebar}
      anchor="left"
    >
      <div className={classes.sidebarHeader}>
        <IconButton onClick={props.toggleSidebar}>
          <ChevronLeft />
        </IconButton>
      </div>
      <Divider />
      <List className={classes.listContainer}>
        <ListItem 
          button
        >
          <ListItemIcon>
            <AssessmentIcon />
          </ListItemIcon>
          <ListItemText>
            Overview
          </ListItemText>
        </ListItem>
        {/* <ListItem 
          button
        >
          <ListItemIcon>
            <MapIcon />
          </ListItemIcon>
          <ListItemText>
            Map
          </ListItemText>
        </ListItem> */}
        <ListItem 
          button
        >
          <ListItemIcon>
            <InfoIcon />
          </ListItemIcon>
          <ListItemText>
            About
          </ListItemText>
        </ListItem>
        {/* <ListItem 
          button
        >
          <ListItemIcon>
            <TimelineIcon />
          </ListItemIcon>
          <ListItemText>
            Smooth data
          </ListItemText>
          <Checkbox
            checked={props.showSmooth}
            onChange={props.toggleSmooth}
            color="default"
          />
        </ListItem> */}
        <ListItem 
          button
        >
          <ListItemIcon>
            <TimelineIcon />
          </ListItemIcon>
          <FormControl 
            className={classes.selectControl}
            variant='outlined'
          >
            <InputLabel 
              variant='outlined'
              id='select-data-range-label'
            >
              Data range
            </InputLabel>
            <Select 
              value={props.viewRange}
              onChange={props.setViewRange}
              autoWidth={true}
              label="Data range"
              labelId='select-data-range-label'
              id='select-data-range'
            >
              <MenuItem value='month'>
                1 month
              </MenuItem>
              <MenuItem value='3month'>
                3 months
              </MenuItem>
              <MenuItem value='all'>
                All data
              </MenuItem>
            </Select>
          </FormControl>
        </ListItem>
      </List>
    </Drawer>
  )
}

const useStyles = makeStyles(theme => ({
  listContainer: {
    minWidth: '240px'
  },
  sidebarHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    padding: theme.spacing(0, 1),
    ...theme.mixins.toolbar
  },
  selectControl: {
    minWidth: 180,
  },
}))

export default Sidebar