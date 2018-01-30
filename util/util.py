from itertools import combinations, chain
import datetime


def get_all_possible_combinations_from_features(data_frame, config):
    primary_feature = config['primary_feature']
    forecast_feature = config['forecast_feature']
    column_combinations = list(data_frame.columns)
    column_combinations = [k for k in column_combinations if (k not in [forecast_feature, primary_feature, 'date'])]
    all_combinations = list(chain(*map(lambda x: combinations(column_combinations, x), range(0, len(column_combinations)+1))))
    final_combinations = []
    for t in all_combinations:
        tmp = list(t)
        tmp.extend([forecast_feature, primary_feature, 'date'])
        final_combinations.append(tmp)
    return final_combinations


def get_final_data_frames_dictionary(original_data_frame, column_combinations):
    key_data_frame = {}
    for combination in column_combinations:
        feature_name = ",".join(combination)
        key_data_frame[feature_name] = {'data_frame': get_data_frame_from_column_names_and_original_data_frame(combination, original_data_frame), 'score': 0}
    return key_data_frame


def get_data_frame_from_column_names_and_original_data_frame(column_names, original_data_frame):
    return original_data_frame[column_names]


def write_results_to_file(results, writer, primary_feature):
    date_forecast_timestamp = results['date_forecast'].tolist()
    date_forecast = list()
    for idx, timestamp in enumerate(date_forecast_timestamp):
        date_forecast.append(datetime.datetime.fromtimestamp(int(timestamp/1000000000)).strftime('%Y-%m-%d'))
    forecast_values = results['forecast'].tolist()
    score = results['score']
    test_size = results['percentage']
    best_features = results['name']
    for index in enumerate(forecast_values):
        writer.writerow([str(primary_feature)]+[best_features]+[date_forecast[index[0]]]+[str(forecast_values[index[0]])]+[str(score)]+[test_size])






