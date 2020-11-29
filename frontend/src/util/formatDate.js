const formatDate = date => {
  const reg = date.matchAll(/([0-9]?[0-9])-([0-9]?[0-9])-([0-9][0-9][0-9][0-9])/gm)
  const match = [...reg][0]
  const month = match[1].length > 1
    ? match[1]
    : `0${match[1]}`
  const day = match[2].length > 1
    ? match[2]
    : `0${match[2]}`
  return `${month}-${day}-${match[3]}`
}

export default formatDate