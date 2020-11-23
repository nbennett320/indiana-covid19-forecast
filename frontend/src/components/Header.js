import React, { useState } from 'react'
import PropTypes from 'prop-types'
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  InputBase,
  useScrollTrigger
} from '@material-ui/core'
import { 
  fade, 
  makeStyles 
} from '@material-ui/core/styles'
import MenuIcon from '@material-ui/icons/Menu'
import SearchIcon from '@material-ui/icons/Search'
import FilterResults from './FilterResults'

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
  const [query, setQuery] = useState('')
  const [menuIsOpen, setMenuIsOpen] = useState(false)
  const anchorRef = React.useRef(null)
  const classes = useStyles()
  const handleQuery = e => {
    setQuery(e.target.value)
    if(query.length > 0 && anchorRef.current) {
      setMenuIsOpen(true)
    }
  }
  const handleClose = e => {
    if(anchorRef.current && anchorRef.current.contains(e.target))
      return
    setMenuIsOpen(false)
  }

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
              aria-label="open drawer"
            >
              <MenuIcon />
            </IconButton>
            <img 
              src="https://upload.wikimedia.org/wikipedia/commons/0/01/Blank_map_subdivisions_Indiana.svg"
              className={classes.indianaIcon}
              alt="Indiana state outline"
            />
            <Typography 
              className={classes.title} 
              variant="h6" 
              color="textPrimary"
              noWrap
            >
              Indiana Covid-19 Forecast
            </Typography>
            <div 
              className={classes.search} 
              ref={anchorRef}
            >
              <div className={classes.searchIcon}>
                <SearchIcon />
              </div>
              <InputBase
                onKeyUp={handleQuery}
                placeholder="Filter"
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
      <FilterResults
        query={query}
        anchorEl={anchorRef.current}
        isOpen={menuIsOpen}
        handleClose={handleClose}
      />
    </div>
  )
}

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
    position: 'fixed',
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
    backgroundColor: fade(theme.palette.primary.dark, 0.25),
    '&:hover': {
      backgroundColor: fade(theme.palette.primary.dark, 0.35),
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
    color: theme.palette.common.white,
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
  indianaIcon: {
    height: '32px',
    padding: '2px 2px',
    paddingRight: '12px',
    transform: 'rotate(6deg)',
    filter: 'hue-rotate(90deg)',
    userSelect: 'none',
  }
}))

export default Header