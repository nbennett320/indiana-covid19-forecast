import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Route, Switch, Redirect } from 'react-router'
import { Footer, Header, Sidebar } from './components/'
import { Summary, About } from './views'
import './assets/css/App.css'

const mql = window.matchMedia(`(max-width: 633px)`)
const viewportWidth = window.innerWidth
const viewportHeight = window.innerHeight

const App = () => {
  const [sidebarIsOpen, setSidebarIsOpen] = React.useState(false)
  const [county, setCounty] = React.useState('Indiana')
  const [showSmooth, setSmooth] = React.useState(false)
  const [smoothingMethod, setSmoothingMethod] = React.useState('polynomial')
  const [showReopeningStages, setShowReopeningStages] = React.useState(false)
  const [showHolidays, setShowHolidays] = React.useState(false)
  const [viewRange, setViewRange] = React.useState('3month')
  const [vpWidth, setViewportWidth] = React.useState(viewportWidth)
  const [vpHeight, setViewportHeight] = React.useState(viewportHeight)

  window.addEventListener('resize', () => {
    setViewportWidth(window.innerWidth)
    setViewportHeight(window.innerHeight)
  })

  const userDevice = {
    isMobile: mql,
    vpWidth,
    vpHeight
  }
  
  return (
    <div style={styles}>
      <Header 
        handleChangeCounty={(val) => setCounty(val)}
        sidebarIsOpen={sidebarIsOpen}
        toggleSidebar={() => setSidebarIsOpen(!sidebarIsOpen)}
        userDevice={userDevice}
      />
      <Sidebar
        isOpen={sidebarIsOpen}
        toggleSidebar={() => setSidebarIsOpen(!sidebarIsOpen)}
        showSmooth={showSmooth}
        toggleSmooth={() => setSmooth(!showSmooth)}
        smoothingMethod={smoothingMethod}
        setSmoothingMethod={setSmoothingMethod}
        showReopeningStages={showReopeningStages}
        toggleReopeningStages={() => setShowReopeningStages(!showReopeningStages)}
        showHolidays={showHolidays}
        toggleHolidays={() => setShowHolidays(!showHolidays)}
        viewRange={viewRange}
        setViewRange={e => setViewRange(e.target.value)}
        userDevice={userDevice}
      />
      <Router>
        <Switch>
          <Route path='/'>
            <Summary 
              county={county}
              showSmooth={showSmooth}
              smoothingMethod={smoothingMethod}
              userDevice={userDevice}
              viewRange={viewRange}
              showReopeningStages={showReopeningStages}
              showHolidays={showHolidays}
            />
          </Route>
          {/* <Route path='/map'>
            <Map />
          </Route> */}
          <Route path='/about'>
            <About />
          </Route>
          <Redirect 
            from='/*'
            to='/'
          />
        </Switch>
      </Router>
      <Footer />
    </div>
  )
}

const styles = {
  height: '100%',
  width: '100%',
}

export default App