import formatDate from './formatDate'
const formatData = (data, viewRange, domainLength,) => {
  if(viewRange.includes('month')) {
    const d = formatDate(
      new Date(
        new Date() - 1000 * 60 * 60 * 24 * domainLength
      ).toLocaleDateString().replaceAll('/','-')
    )
    return data.filter(el => {
      const elArr = el.x.split('-').map(s => Number(s))
      const dArr = d.split('-').map(s => Number(s))
      if(elArr[2] < dArr[2])
        return false
      else if(elArr[0] < dArr[0])
        return false
      else if((elArr[0] === dArr[0]) && (elArr[1] < dArr[1]))
        return false
      else return true
    })
  } else return data
}

export default formatData