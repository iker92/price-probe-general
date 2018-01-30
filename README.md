# price-probe-general purpose

General purpose ARIMA algorithm. With your time series data.

## How to make it work
- Install all dependencies via `pip install -r requirements.txt`
Edit your `conf/conf.json` like:
- Primary feature, if not present. It's required to have a primary key, in our case `item` in order to make the algorithm work.
- Forecast feature, column name of the value that will be forecasted
- Size of the test_set, e.g. 10%

The date format must be like `YYYY-MM-DD`
