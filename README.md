# price-probe-general purpose

General purpose ARIMA algorithm. With your time series data, you have to specify in conf/conf.json :

-The primary feature, if not present beacause the training set is all about one thing, add another column in csv file with a name in all rows <br>
-The forecast feature, column name of the value that will be forecasted <br>
-The size of the test_set, e.g. 10% <br>
The date information must be like 'YYYY-MM-DD'
