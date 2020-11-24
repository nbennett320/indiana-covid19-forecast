import React from 'react'
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Divider,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import ChevronLeft from '@material-ui/icons/ChevronLeft'
import TimelineIcon from '@material-ui/icons/Timeline'
import MapIcon from '@material-ui/icons/Map'
import InfoIcon from '@material-ui/icons/Info'

const Sidebar = props => {
  const classes = useStyles()
  return (
    <Drawer
      open={props.isOpen}
      variant="persistent"
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
            <TimelineIcon />
          </ListItemIcon>
          <ListItemText>
            Overview
          </ListItemText>
        </ListItem>
        <ListItem 
          button
        >
          <ListItemIcon>
            <MapIcon />
          </ListItemIcon>
          <ListItemText>
            Map
          </ListItemText>
        </ListItem>
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
}))

export default Sidebar