import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Route, Switch, Redirect } from 'react-router'
import { Header, Sidebar } from './components/'
import { Map, Summary, About } from './views'
import './assets/css/App.css'

const App = () => {
  const [sidebarIsOpen, setSidebarIsOpen] = React.useState(false)
  const [county, setCounty] = React.useState('Indiana')
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
      />
      <Router>
        <Switch>
          <Route path='/' exact>
            <Summary 
              county={county}
            />
          </Route>
          <Route path='/map' exact>
            <Map />
          </Route>
          <Route path='/about' exact>
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