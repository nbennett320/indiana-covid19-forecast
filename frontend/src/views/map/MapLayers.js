import ReactMapboxGl, { Layer, Feature } from 'react-mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

const Map = ReactMapboxGl({
  accessToken: 'pk.eyJ1IjoibmJlbm5ldHQzMjAiLCJhIjoiY2tndm96MzByMDBwcTJzbnhlcTNvb2xrMSJ9.FIOqlHgzOFa2u5wZKc4jXA'
})

const MapLayers = props => {
  return (
    <Map
      style="mapbox://styles/mapbox/streets-v9"
      containerStyle={{
        height: '100%',
        width: '100%'
      }}
    >
      <Layer 
        type="symbol" 
        id="marker" 
        layout={{ 'icon-image': 'marker-15' }}
      >
        <Feature coordinates={[-86.15804, 39.76838]} />
      </Layer>
    </Map>
  )
}

export default MapLayers