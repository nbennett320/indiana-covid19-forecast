import { formatDate } from './'

const getRenderedData = props => {
  const rawData = props.isCountyLevelData 
    ? require(`../data/model_prediction_${props.county}_${props.plotData}.json`)
    : require(`../data/model_prediction_${props.plotData}.json`)
  const {
    x_data,
    y_data,
    x_pred,
    y_pred,
    x_data_polynomial,
    y_data_polynomial,
    x_pred_polynomial,
    y_pred_polynomial,
    model_result,
    prediction_key
  } = rawData
  x_data.map((t, i) => ({
    x: new Date(t / Math.pow(10, 6)),
    y: y_data[i],
    name: props.format.dataLab
  }))
  const predictionKey = String(prediction_key)
  const calcError = predictionKey.includes('hospital')
    ? predictionKey.includes('vent') 
      ? (i, sampleSize) => ((model_result['prediction/mean'] / model_result['label/mean']) * (i)) / sampleSize
      : (i, sampleSize) => ((model_result['prediction/mean'] * (i / (sampleSize - 1))) / sampleSize) * (model_result['prediction/mean'] / model_result['label/mean'])
    : (i, sampleSize) => (model_result['prediction/mean'] * (i)) / sampleSize
  const concatErrorMargins = (val, i, sampleSize) => [
    val - calcError(i, sampleSize),
    val + calcError(i, sampleSize)
  ]
  const data = []
  if(!props.showSmooth) {
    data.push(...[
      ...x_data.map((t, i) => ({
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        y_all: y_data[i],
        name: props.format.dataLab
      })),
      ...[x_data[x_data.length - 1], x_pred[0]].map((t, i) => ({
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        y_all: Math.round([y_data[y_data.length - 1], y_pred[0]][i]),
        y_pred: concatErrorMargins(Math.round([y_data[y_data.length - 1], y_pred[0]][i]), i, y_pred.length),
        name: 'Today'
      })),
      ...x_pred.map((t, i) => ({
        x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
        y_all: Math.round(y_pred[i]),
        y_pred: concatErrorMargins(Math.round(y_pred[i]), i + 2, y_pred.length),
        name: 'Forecasted'
      })),
    ])
  }
  if(props.showSmooth) {
    if(props.smoothingMethod === 'polynomial') {
      data.push(...[
        ...x_data_polynomial.map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          y_all: y_data_polynomial[i],
          name: props.format.dataLab
        })),
        ...[x_data_polynomial[x_data_polynomial.length - 1], x_pred_polynomial[0]].map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          y_all: [y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]][i],
          y_pred: concatErrorMargins([y_data_polynomial[y_data_polynomial.length - 1], y_pred_polynomial[0]][i], y_data_polynomial.length),
          name: 'Today'
        })),
        ...x_pred_polynomial.map((t, i) => ({
          x: formatDate(new Date(t / Math.pow(10, 6)).toLocaleDateString().replaceAll('/','-')),
          y_all: y_pred_polynomial[i],
          y_pred: concatErrorMargins(y_pred_polynomial[i], i + 2, y_pred_polynomial.length),
          name: 'Forecasted'
        }))
      ])
    }
  }
  return {
    data: data.flat(),
    model_result: model_result
  }
}

export default getRenderedData