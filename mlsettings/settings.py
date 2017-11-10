import os
import sys
import json

APP_ML_PATH = "ML_PATH"
APP_ML_DATA_PATH = "ML_DATASOURCE"

DEFAULT_FILENAME = r"F:\MachineLearning\mlsettings\mlconfig.json"
DEFAULT_APP_ENV = "DEV"
CONFIG_KEYS = ['ML_DATASOURCE', 'ML_PATH']


def load_app_config(filename=DEFAULT_FILENAME, app_env=DEFAULT_APP_ENV):
    config_dict = load_config_json(filename)
    print(config_dict)
    for config_value in config_dict[app_env]:

        if config_value in os.environ:
            sys.path.append(os.environ.get(config_value))
            print("Adding {0}  to system path".format(os.environ.get(config_value)))
        else:
            print("{0} not found".format(config_value))


def load_config_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as logging_configuration_file:
            config_dict = json.load(logging_configuration_file)
        return config_dict
    else:
        return None


def get_datafolder_path(data_path=APP_ML_DATA_PATH):
    if data_path in os.environ:
        return os.environ.get(data_path)
    else:
        return os.environ(APP_ML_DATA_PATH)


def load_ml_path():
    if APP_ML_PATH in os.environ:
        sys.path.append(os.environ.get('ML_PATH'))
        print("Adding {0}  to system path".format(os.environ.get('ML_PATH')))
        return True
    else:
        print("{0} not found".format(APP_ML_PATH))
        sys.exit(-1)
        return False
