import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Route, Switch, Redirect } from 'react-router'
import { Header, Sidebar } from './components/'
import { Map, Summary, About } from './views'
import './assets/css/App.css'

const App = () => {
  const [sidebarIsOpen, setSidebarIsOpen] = React.useState(false)
  const [county, setCounty] = React.useState('Indiana')
  const [showSmooth, setSmooth] = React.useState(false)
  const [smoothingMethod, setSmoothingMethod] = React.useState('polynomial')
  return (
    <div style={styles}>
      <Header 
        handleChangeCounty={(val) => setCounty(val)}
        sidebarIsOpen={sidebarIsOpen}
        toggleSidebar={() => setSidebarIsOpen(!sidebarIsOpen)}
      />
      <Sidebar
        isOpen={sidebarIsOpen}
        toggleSidebar={() => setSidebarIsOpen(!sidebarIsOpen)}
        showSmooth={showSmooth}
        toggleSmooth={() => setSmooth(!showSmooth)}
        smoothingMethod={smoothingMethod}
        setSmoothingMethod={setSmoothingMethod}
      />
      <Router>
        <Switch>
          <Route path='/'>
            <Summary 
              county={county}
              showSmooth={showSmooth}
              smoothingMethod={smoothingMethod}
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
    </div>
  )
}

const styles = {
  height: '100vh',
  width: '100%',
}

export default App