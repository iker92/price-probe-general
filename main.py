import os
import warnings
import csv

import pandas as pd
from conf import configuration
import arima
from util import util

warnings.filterwarnings("ignore")
config = configuration.get_configuration()
path = os.path.dirname(os.path.abspath(__file__))

training = pd.read_csv("/{}/resource/data.csv".format(path))

primary_feature = config['primary_feature']
forecast_feature = config['forecast_feature']
entries = pd.unique(training[primary_feature])
file = "/{}/resource/results.csv".format(path)
with open(file, 'w') as outcsv:
    writer = csv.writer(outcsv, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Primary Feature", "Best features", "Date", "Forecast", "Score", "Test size"])
    for entry in entries:
        results = {}
        df_filtered = training.loc[training[primary_feature] == entry]
        features_combination = util.get_all_possible_combinations_from_features(df_filtered, config)
        final_dictionary_data_frame = util.get_final_data_frames_dictionary(df_filtered, features_combination)
        arima_dict = arima.find_arima_parameters_by_dataframe(df_filtered, forecast_feature)
        for attr, value in final_dictionary_data_frame.items():
            arima_dict['data_frame'] = value['data_frame']
            results[attr] = arima.test_arima(attr, arima_dict, config)
        best_result = arima.find_best_result(results)
        util.write_results_to_file(best_result, writer, df_filtered[primary_feature].values[0])
