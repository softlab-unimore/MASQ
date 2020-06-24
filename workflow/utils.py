import pandas as pd
from dbconnection.connector import get_table
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def get_dataframe(file):
    try:
        ds = pd.read_csv(file, sep=',')
    except (pd.errors.ParserError, ValueError):
        return None

    return ds


def get_dataset(scenario):
    if scenario['is_db']:
        ds = get_table(scenario['db_url'], scenario['table'])
        if ds is None:
            raise ValueError('Impossible read dataset from DBMS, please check DBMS input')

    else:
        ds = get_dataframe(scenario['dataset'])
        if ds is None:
            raise ValueError('Impossible read dataset from selected file')

    return ds


def get_train_test(ds, labels, val):
    # sep = round(len(ds) * (1 - (val / 100)))
    #
    # x_train = ds.iloc[:sep, :]
    # y_train = labels[:sep]
    #
    # x_test = ds.iloc[sep:, :]
    # y_test = labels[sep:]

    x_train, x_test, y_train, y_test = train_test_split(ds, labels, test_size = val / 100, random_state = 42)

    return x_train, y_train, x_test, y_test


def get_sql_predictions_effectiveness(sklearn_preds_file, sql_preds_file):

    sklearn_preds = pd.read_csv(sklearn_preds_file).iloc[:,0].to_list()
    sql_preds = pd.read_csv(sql_preds_file).iloc[:,0].to_list()

    res = r2_score(sklearn_preds, sql_preds)
    return res
