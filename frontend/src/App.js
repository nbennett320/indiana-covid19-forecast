import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Route, Switch } from 'react-router'
import { Header } from './components/'
import { Map, Summary, About } from './views'
import './assets/css/App.css'

const App = () => {
  const [county, setCounty] = React.useState('Indiana')

  return (
    <div style={styles}>
      <Header 
        handleChangeCounty={(val) => setCounty(val)}
      />
      <Router>
        <Switch>
          <Route path='/'>
            <Summary 
              county={county}
            />
          </Route>
          <Route path='/map'>
            <Map />
          </Route>
          <Route path='/about'>
            <About />
          </Route>
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