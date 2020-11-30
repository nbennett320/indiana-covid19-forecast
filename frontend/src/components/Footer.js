import React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import {
  Divider,
  Link,
  Icon
} from '@material-ui/core'
import GitHubIcon from '@material-ui/icons/GitHub'

const Footer = () => {
  const classes = useStyles()
  return (
    <div className={classes.main}>
      <Divider light/>
      <div className={classes.linkContainer}>
        <Icon className={classes.icon}>
          <GitHubIcon />
        </Icon>
        <Link
          href='https://github.com/nbennett320/indiana-covid19-forecast'
          className={classes.link}
          variant='body2'
          color='textSecondary'
          underline='hover'
        >
          View on GitHub
        </Link>
      </div>
    </div>
  )
}

const useStyles = makeStyles(() => ({
  main: {
    width: '90%',
    minHeight: '64px',
    backgroundColor: '#fcfcfc',
    padding: '24px 5%'
  },
  linkContainer: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    padding: '16px 24px',
    marginTop: '6px'
  },
  link: {
    marginLeft: '12px',
    color: 'rgba(0,0,0,0.44)'
  },
  icon: {
    color: 'rgba(0,0,0,0.44)'
  }
}))

export default Footer