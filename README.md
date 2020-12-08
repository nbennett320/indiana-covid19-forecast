## Indiana Covid-19 Forecast

One-week forecasting for Covid-19 cases, deaths, hospital occupancy, and ventilator availability in Indiana.

### Data sources
[Mobility trends from Apple](https://covid19.apple.com/mobility) <br />
[Mobility trends from Google](https://www.google.com/covid19/mobility/) <br />
[Time series data from Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/Covid-19) <br />
[2019 population & housing unit estimates from the US Census Bureau](https://www.census.gov/programs-surveys/popest.html) <br />
[2014—2018 American Community Surveys, 5-Year Data Profiles from the US Census Bureau](https://www.census.gov/acs/www/data/data-tables-and-tools/data-profiles/2018/) <br />
[Building permits from the US Census Bureau](https://www.census.gov/construction/bps/) <br />
[Household size by county from the US Census Bureau](https://www.census.gov/topics/families.html) <br />
[Computer and internet use from the US Census Bureau](https://www.census.gov/topics/population/computer-internet.html) <br />
[Indiana age and sex data from the US Census Bureau](https://www.census-charts.com/ASC/Indiana.html) <br />
[Indiana county-wide Covid-19 testing, case, and death trends from the Indiana Data Hub](https://hub.mph.in.gov/dataset/covid-19-county-wide-test-case-and-death-trends/resource/afaa225d-ac4e-4e80-9190-f6800c366b58) <br />
[Indiana Covid-19 case demographics by county and district data from the Indiana Data Hub](https://hub.mph.in.gov/dataset/covid-19-case-demographics-by-county/resource/9ae4b185-b81d-40d5-aee2-f0e30405c162) <br />
[Indiana Covid-19 deaths by age group data from the Indiana Data Hub](https://hub.mph.in.gov/dataset/covid-19-deaths-by-date-by-age-group) <br />
[Indiana Covid-19 cases by school data from the Indiana Data Hub](https://hub.mph.in.gov/dataset/covid-19-cases-by-school/resource/39239f34-11ff-4dfc-9b9a-a408b0399458) <br />
[Indiana bed availability by date from the Indiana Data Hub](https://hub.mph.in.gov/dataset/covid-19-bed-and-vent-usage-by-day/resource/0c00f7b6-05b0-4ebe-8722-ccf33e1a314f) <br />
[Indiana prevention measures and reopening stages data from www.backontrack.in.gov](https://www.backontrack.in.gov/2348.htm)

### Usage
- [General](#general)
- [Python scripts](#python_scripts)
- [Frontend scripts](#frontend_scripts)

<a name="general"></a>

#### General

**`make update-model`** <br/>
Update datasets and county-level models.
<br/>

**`make update-frontend`** <br/>
Rebuild and deploy frontend.
<br/>

**`make install-model-dependencies`** <br/>
Install dependencies for compiling new models.
<br/>

**`make clean`** <br/>
Remove temp directories.
<br/>

<a name="python_scripts"></a>

#### Python scripts

**`python3 scripts/model.py`**<br/>
Rebuild models. <br/>
Flags:
- `-d`
- `--days`
  - Number of days to forecast predictions for.
  - _Default: 14_
  - _Type: int_
- `-C`
- `--county`
  - Specific county to generate model for. `Indiana` generates state-level predictions, `All` generates predictions for all counties.
  - _Default: Indiana_
  - _Type: str_
- `-D`
- `--train-dir`
  - Output directory for model files.
  - _Default: `train/`_
  - _Type: str_
- `-o`
- `--output-dir`
  - Output directory for data files preformatted for the frontend.
  - _Default: `frontend/src/data/`_
  - _Type: str_
- `-u`
- `--update-datasets`
  - Update datasets.
  - _Default: `False`_
  - _Type: bool_
- `-v`
- `--verbose`
  - Use verbose console messages.
  - _Default: `False`_
  - _Type: bool_
- `-P`
- `--plot`
  - Plot predictions for model being generated.
  - _Default: `False`_
  - _Type: bool_
<br/>

<a name="frontend_scripts"></a>

#### Frontend scripts

**`yarn serve`** <br/>
Host a development server at [localhost:3000/](localhost:3000/).
<br/>

**`yarn build`** <br/>
Build production bundle to `frontend/build`.
<br/>

**`yarn deploy`** <br/>
Build and deploy to [GitHub Pages](https://nbennett320.github.io/indiana-covid19-forecast/).
<br/>