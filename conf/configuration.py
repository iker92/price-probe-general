import json


def get_configuration():
    data = json.load(open('conf/conf.json'))
    test_size = float(data['Test_size'][0:1]) / 10
    primary_feature = data['Primary_feature']
    forecast_feature = data['Forecast_feature']

    return {'test_size': test_size,
            'primary_feature': primary_feature,
            'forecast_feature': forecast_feature
            }
