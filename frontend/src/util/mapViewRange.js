const mapViewRange = str => {
  switch(str) {
    case 'month':
      return 'past 31 days'
    case '3month':
      return 'past 3 months'
    default:
      return 'all time'
  }
}

export default mapViewRange