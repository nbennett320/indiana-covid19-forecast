### Indiana Covid-19 Forecast &ndash; Scripts

Scripts used to operate on data and build prediction models.

#### Usage

**`model.py`**<br/>
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
- `-S`
- `--smooth-mode`
  - Show smooth curves if predictions are being plotted. Valid interpolation methods are `polynomial` or `spline`.
  - _Default: `False`_
  - _Type: bool_
<br/>