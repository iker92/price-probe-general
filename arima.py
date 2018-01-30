
# 3rd party
import numpy
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, adfuller
from conf import configuration

config = configuration.get_configuration()
test_size = config['test_size']


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * (1 - test_size))
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=-1)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), (0, 0, 0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    return best_cfg


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def find_d(series, threshold, d):
    adfuller_test_results = adfuller(series.values)
    if (adfuller_test_results[0] < 0 and adfuller_test_results[4]['5%'] > adfuller_test_results[0] and adfuller_test_results[0] < threshold) or d > 2:
        return d
    else:
        series = series.diff()
        series = series.dropna()
        d += 1
        return find_d(series, threshold, d)


def test_arima(title, arima_dict, config):
    results = {}
    forecast = None
    primary_feature = config['primary_feature']
    forecast_feature = config['forecast_feature']
    forecast_feature_df = arima_dict['forecast_feature_df']
    data_frame = arima_dict['data_frame']
    p, d, q = arima_dict['order']
    data_frame['date'] = pd.to_datetime(data_frame['date'])
    # Setting index on date
    data_frame = data_frame.set_index('date')
    best_configuration = evaluate_models(forecast_feature_df, range(0, p), [d], range(0, q))
    size = int(len(forecast_feature_df) * (1 - test_size))
    training_set, test = forecast_feature_df[0:size], forecast_feature_df[size:]
    # External Columns
    features_names = title.split(',')
    print(best_configuration)
    print(title)
    if len(features_names) > 3:
        exogenous_features_names = [k for k in features_names if (k not in [primary_feature, forecast_feature, 'date'])]
        not_unique_exogenous_features = pd.Series()
        selected_exogenous_features = data_frame[exogenous_features_names]
        for name in selected_exogenous_features.columns:
            if not filter_unique_values(selected_exogenous_features[name]):
                not_unique_exogenous_features = not_unique_exogenous_features.append(selected_exogenous_features[name])

        if not_unique_exogenous_features.size > 0:
            exogenous_features, exogenous_features_test = not_unique_exogenous_features[0:size], not_unique_exogenous_features[size:]
            try:
                model = ARIMA(training_set, order=best_configuration, exog=exogenous_features)
                model_fit = model.fit(disp=-1)
                forecast = model_fit.forecast(steps=len(test), exog=exogenous_features_test)[0]
            except:
                pass
    else:
        try:
            # Fit model
            model = ARIMA(training_set, order=best_configuration)
            model_fit = model.fit(disp=-1)
            # one-step out-of sample forecast
            forecast = model_fit.forecast(steps=len(test))[0]
        except:
            pass

    date_forecast = test.index._data
    if forecast is not None:
        if False in np.isnan(forecast):
            results[title] = {
                'forecast': forecast, 'date_forecast': date_forecast,
                'score': mean_absolute_percentage_error(test.values, forecast), 'values': forecast_feature_df,
                'training_set': training_set,
            }
    return results


def find_best_result(results):
    best_score = 0

    best_result = {}
    for attr, value in results.items():
        if attr in results[attr]:
            if best_score == 0 or results[attr][attr]['score'] < best_score:
                best_score = results[attr][attr]['score']
                best_result = {
                    'name': attr,
                    'values': results[attr][attr]['values'],
                    'training_set': results[attr][attr]['training_set'],
                    'date_forecast': results[attr][attr]['date_forecast'],
                    'forecast': results[attr][attr]['forecast'],
                    'score': best_score,
                    'percentage': str(int(config['test_size'] * 100)) + "%"
                }
    return best_result


def find_p_d_q_values(prices_column, prices_elements_number):

    # First time a ACF value crosses positive threshold (AR)
    p = 0
    # Number of times needed to make the series stationary (I)
    d = find_d(prices_column, 0.05, 0)
    # First time a PACF value crosses positive threshold (MA)
    q = 0

    p, q = compute_acf_pacf(prices_column, prices_elements_number)

    return p, d, q


def compute_acf_pacf(prices_column, prices_elements_number):
    # ACF = Autocorrelation Function. How big are clusters of data containing elements with similar trend.
    # https://onlinecourses.science.psu.edu/stat510/node/60
    # Threshold values for ACF and PACF for 95% Confidence Interval
    positive_threshold = 1.96 / np.sqrt(prices_elements_number)
    if prices_column.min() == prices_column.max():
        return 0, 0
    lag_acf = acf(prices_column, nlags=prices_elements_number)
    # PACF = Partially Autocorrelation Function. Correlation between points not looking at already visited ones.
    # https://onlinecourses.science.psu.edu/stat510/node/46
    lag_pacf = pacf(prices_column, nlags=prices_elements_number)
    acf_it = 0
    pacf_it = 0
    for value in np.nditer(lag_acf):
        if value < positive_threshold:
            q = acf_it
            if value < -positive_threshold:
                q += 1
            break
        else:
            acf_it += 1

    for value in np.nditer(lag_pacf):
        if value < positive_threshold:
            p = pacf_it
            if value < -positive_threshold:
                p += 1
            break
        else:
            pacf_it += 1
    return p, q


# this method computes acf and pacf for given series and returns upper bound values from p,d,q parameters for arima
def find_arima_parameters_by_dataframe(data_frame, forecast_feature):
    first_configuration = {}
    data_frame['date'] = pd.to_datetime(data_frame['date'])

    # Setting index on date
    data_frame = data_frame.set_index('date')

    forecast_feature_df = data_frame[forecast_feature]
    forecast_feature_elements_number = len(forecast_feature_df)

    p, d, q = find_p_d_q_values(forecast_feature_df, forecast_feature_elements_number)
    first_configuration['order'] = p, d, q
    first_configuration['forecast_feature_df'] = forecast_feature_df
    return first_configuration


# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
def mean_absolute_percentage_error(test, forecast):
    size = len(test)
    m = (100 / size)
    abs_sum = 0.0
    for i in range(len(test)):
        if test[i] == 0:
            size -= 1
            m = (100 / size)
            continue
        abs_sum += abs((test[i] - forecast[i]) / test[i])
    return abs_sum * m


def check_unique_values(exogenous_features_names, exogenous_features):

    results = {}
    duplicated = False
    count = 0
    check_duplicated = pd.Series()
    if len(exogenous_features_names) > 0:
        for name in exogenous_features_names:
            if name in exogenous_features.columns:
                check_duplicated = exogenous_features.duplicated(subset=name, keep=False)
            if check_duplicated.size > 0:
                for value in np.nditer(check_duplicated.values):
                    if value:
                        count += 1
            if count == check_duplicated.size:
                duplicated = True
            results['name'] = name
            results['name']['duplicated'] = duplicated
    return results


def filter_unique_values(features):
    first_value = features.values[0]
    if not first_value and first_value != 0.0:
        return True
    list = features.tolist()
    result = [value for value in list if value == first_value ]
    if result == features.size:
        return True
    else:
        return False






