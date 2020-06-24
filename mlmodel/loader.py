import logging
from joblib import dump, load
from pickle import UnpicklingError, PickleError


def load_model(file):
    try:
        model = load(file)
        return model
    except UnpicklingError as e:
        logging.error(e)
        return None


def save_model(model, name):
    try:
        dump(model, name)
    except PickleError as e:
        logging.error(e)
        return None
