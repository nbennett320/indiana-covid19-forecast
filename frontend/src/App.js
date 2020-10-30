import React from 'react'
import { Router, Route, Switch } from "react-router"
import { Header } from './components/'
import { Map } from './views'
import './assets/css/App.css'

const App = () => {
  return (
    <div style={styles}>
      <Header />
      <Router>
        <Switch>
          <Route path='/'>
            
          </Route>
          <Route path='/map'>
            <Map />
          </Route>
          <Route path='/about'>
            <Map />
          </Route>
        </Switch>
      </Router>
    </div>
  )
}

const styles = {
  height: '100vh',
  width: '100vw'
}

export default App