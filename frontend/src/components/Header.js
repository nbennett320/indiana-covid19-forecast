import React from 'react'
import PropTypes from 'prop-types'
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  InputBase,
  SvgIcon,
  useScrollTrigger
} from '@material-ui/core'
import { 
  fade, 
  makeStyles 
} from '@material-ui/core/styles'
import MenuIcon from '@material-ui/icons/Menu'
import SearchIcon from '@material-ui/icons/Search'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    position: 'absolute',
    zIndex: '100',
    width: '100%'
  },
  menuButton: {
    marginRight: theme.spacing(2),
  },
  title: {
    flexGrow: 1,
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'block',
    },
  },
  search: {
    position: 'relative',
    borderRadius: theme.shape.borderRadius,
    backgroundColor: fade(theme.palette.common.black, 0.15),
    '&:hover': {
      backgroundColor: fade(theme.palette.common.black, 0.25),
    },
    marginLeft: 0,
    width: '100%',
    [theme.breakpoints.up('sm')]: {
      marginLeft: theme.spacing(1),
      width: 'auto',
    },
  },
  searchIcon: {
    padding: theme.spacing(0, 2),
    height: '100%',
    position: 'absolute',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  inputRoot: {
    color: 'inherit',
  },
  inputInput: {
    padding: theme.spacing(1, 1, 1, 0),
    paddingLeft: `calc(1em + ${theme.spacing(4)}px)`,
    transition: theme.transitions.create('width'),
    width: '100%',
    [theme.breakpoints.up('sm')]: {
      width: '12ch',
      '&:focus': {
        width: '20ch',
      },
    },
  },
}))

const ElevationScroll = props => {
  const { children, window } = props
  const trigger = useScrollTrigger({
    disableHysteresis: true,
    threshold: 0,
    target: window ? window() : undefined,
  })

  return React.cloneElement(children, {
    elevation: trigger ? 4 : 0,
  })
}

ElevationScroll.propTypes = {
  children: PropTypes.element.isRequired,
  window: PropTypes.func,
}

const Header = props => {
  const classes = useStyles()
  return (
    <div className={classes.root}>
      <ElevationScroll {...props}>
        <AppBar 
          position="static" 
          className="header"
        >
          <Toolbar>
            <IconButton
              edge="start"
              className={classes.menuButton}
              color="textSecondary"
              aria-label="open drawer"
            >
              <MenuIcon />
            </IconButton>
            <img 
              src="https://upload.wikimedia.org/wikipedia/commons/0/01/Blank_map_subdivisions_Indiana.svg"
              style={styles.indianaIcon}
            />
            <Typography 
              className={classes.title} 
              variant="h6" 
              color="textPrimary"
              noWrap
            >
              Indiana Covid-19 Forecast
            </Typography>
            <div className={classes.search}>
              <div className={classes.searchIcon}>
                <SearchIcon />
              </div>
              <InputBase
                placeholder="Search…"
                color="textSecondary"
                classes={{
                  root: classes.inputRoot,
                  input: classes.inputInput,
                }}
                inputProps={{ 'aria-label': 'search' }}
              />
            </div>
          </Toolbar>
        </AppBar>
      </ElevationScroll>
    </div>
  )
}

const styles = {
  indianaIcon: {
    height: '32px',
    padding: '2px 2px',
    paddingRight: '8px',
    transform: 'rotate(6deg)',
    filter: 'hue-rotate(90deg)',
    userSelect: 'none',
  }
}

export default Header