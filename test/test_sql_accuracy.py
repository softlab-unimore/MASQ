import os
import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array
from sklearn.tree._tree import DTYPE

from dbconnection.connector import get_connector, get_tables
from mlmodel.mlmanager import MLManager, Optimizer
from test.check_tree import check_path
from test.confs import BikeConf, CreditCardConf, HeartConf, CriteoConf, FlightConf, IrisConf, TaxiConf
from utils.ml_eval import evaluate_binary_classification_results, evaluate_multi_classification_results, \
    evaluate_regression_results


PROJECT_DIR = os.path.abspath('..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'test_data')


def check_data_config(conf: dict):
    assert isinstance(conf, dict)

    params = conf.copy()

    conf_params = ['use_case', 'train_file', 'test_file', 'label', 'str_db_conn', 'train_table_name', 'test_table_name']
    for conf_param in conf_params:
        assert conf_param in conf

    USE_CASE_DIR = os.path.join(DATA_DIR, conf['use_case'])
    assert os.path.exists(USE_CASE_DIR)

    train_file = os.path.join(USE_CASE_DIR, conf['train_file'])
    assert os.path.exists(train_file)
    train = pd.read_csv(train_file)

    test_file = os.path.join(USE_CASE_DIR, conf['test_file'])
    assert os.path.exists(test_file)
    test = pd.read_csv(test_file)

    label = conf['label']
    assert label in train.columns
    assert label in test.columns
    assert list(train.columns) == list(test.columns)
    y_train = train[label]
    train_with_label = train.copy()
    train = train.drop([label], axis=1)
    y_test = test[label]
    test_with_label = test.copy()
    test = test.drop([label], axis=1)

    params['train'] = train
    params['y_train'] = y_train
    params['test'] = test
    params['y_test'] = y_test

    conn = get_connector(conf['str_db_conn'])
    assert conn is not None
    params['db_conn'] = conn

    tables = get_tables(conf['str_db_conn'])
    assert tables is not None

    train_table = conf['train_table_name']
    if train_table in tables:
        logging.warning(f"Table {train_table} already exists.")
    else:
        logging.warning(f"Starting creating the table {train_table}...")
        train_with_label.to_sql(train_table, conn, index=False)
        logging.warning(f"Table {train_table} created successfully.")

    test_table = conf['test_table_name']
    if test_table in tables:
        logging.warning(f"Table {test_table} already exists.")
    else:
        logging.warning(f"Starting creating the table {test_table}...")
        test_with_label.to_sql(test_table, conn, index=False)
        logging.warning(f"Table {test_table} created successfully.")

    return params


def check_pipeline_config(pipeline_config: dict, available_features: list):
    assert isinstance(pipeline_conf, dict)
    assert isinstance(available_features, list)
    assert len(available_features) > 0

    assert 'model' in pipeline_config
    assert 'transforms' in pipeline_config

    model = pipeline_config['model']
    assert isinstance(model, dict)
    assert 'name' in model
    assert isinstance(model['name'], str)
    assert 'obj' in model
    assert model['obj'] is not None

    transforms = pipeline_config['transforms']
    for transform in transforms:
        assert isinstance(transform, dict)
        assert 'name' in transform
        assert isinstance(transform['name'], str)
        assert 'obj' in transform
        assert transform['obj'] is not None
        assert 'features' in transform
        assert isinstance(transform['features'], list)
        assert len(transform['features']) > 0
        for f in transform['features']:
            assert isinstance(f, str)
            assert f in available_features


def create_pipeline(pipeline_conf: dict):
    assert isinstance(pipeline_conf, dict)

    pipeline_transforms = []
    for transform in pipeline_conf['transforms']:
        pipeline_transforms.append(
            (transform['name'], transform['obj'], transform['features'])
        )

    pipeline_transforms = ('pipeline_transforms',
                           ColumnTransformer(remainder='passthrough',
                                             transformers=pipeline_transforms))

    pipeline_estimator = (pipeline_conf['model']['name'], pipeline_conf['model']['obj'])

    pipeline = Pipeline(steps=[pipeline_transforms, pipeline_estimator])
    return pipeline


def extract_pipeline(pipeline):
    model_name, trained_model = pipeline.steps[1]
    transforms = []
    for idx in range(len(pipeline.steps[0][1].transformers)):
        a, b, c = pipeline.steps[0][1].transformers_[idx]
        transforms.append({
            'transform_name': a,
            'fitted_transform': b,
            'transform_features': c,
        })

    return {
        'model': {
            'model_name': model_name,
            'trained_model': trained_model
        },

        'transforms': transforms
    }


def create_query(pipeline, mlmanager, features, input_table, optimizer, debug=False):
    opt = Optimizer(pipeline, features, optimizer)
    pipeline = opt.optimize()

    # create an SQL query for each transformer
    sql_transformers = pipeline["transforms"]
    for sql_transformer in sql_transformers:
        transformation_query = sql_transformer.query(input_table)

        # the input table for the possible next transformer is the output of the current transformer
        input_table = "({}) AS data".format(transformation_query)

    # create the SQL query that performs the model inference
    model = pipeline['model']
    # model_name = model["model_name"]
    fitted_model = model["trained_model"]
    # model_sql_wrapper = self.sql_model_types[model_name]
    model_sql_wrapper = model['model_sql_wrapper']
    assert 'model_features' in model
    if model['model_features'] is not None:
        features = model['model_features']

    if debug:
        print(type(fitted_model), '->', type(model_sql_wrapper))
        for attribute, value in model_sql_wrapper.__dict__.items():
            print("\t", attribute, '=', value)

    query = model_sql_wrapper.query(fitted_model, features, input_table)

    return query


def _check_fitted_pipeline(pipeline, model_name, X):

    # Sklearn's GradientBoostingClassifier adds to the final score (i.e., the weighted sum of the tree scores) an
    # init score.
    # This part has not been implemented in SQL, but this score has been added to the final query as an offset.

    # retrieve the Sklearn's GradientBoostingClassifier init score
    if model_name.startswith("GradientBoosting"):
        model = pipeline[model_name]

        transformers = pipeline["pipeline_transforms"]
        transformed_data = transformers.transform(X)

        transformed_data = check_array(transformed_data, dtype=DTYPE, order="C", accept_sparse='csr')
        init_score = model._raw_predict_init(transformed_data).ravel()[0]
        model.init_score = init_score


def main(data_conf, pipeline_conf, task, debug=False):

    data_conf['str_db_conn'] = 'mysql+pymysql://user:password@localhost/MASQdb'
    data_conf = check_data_config(data_conf)
    train = data_conf['train']
    y_train = data_conf['y_train']
    test = data_conf['test']
    y_test = data_conf['y_test']
    test_table_name = data_conf['test_table_name']
    features = list(data_conf['train'].columns)
    conn = data_conf['db_conn']

    tasks = ['regression', 'binary_classification', 'multi_classification']
    if task not in tasks:
        raise ValueError(f"Wrong task {task}. Available tasks: {tasks}")

    if task == 'regression':
        eval_fn = evaluate_regression_results
    elif task == 'binary_classification':
        eval_fn = evaluate_binary_classification_results
    else:
        eval_fn = evaluate_multi_classification_results

    check_pipeline_config(pipeline_conf, features)
    model_name = pipeline_conf['model']['name']

    mlmanager = MLManager()
    pipeline = create_pipeline(pipeline_conf)

    # fit
    print("\nStarting training...")
    pipeline.fit(train, y_train)
    _check_fitted_pipeline(pipeline, model_name, train)

    print("Training completed.\n")

    fitted_model = pipeline.steps[1][1]

    # ML predict
    print("\nStarting the ML inference...")
    ml_preds = pipeline.predict(test)
    ml_preds = pd.Series(ml_preds)
    print(ml_preds[:10])
    eval_fn(model_name, y_test, ml_preds)
    print("ML inference completed.\n")

    # SQL conversion
    print("\nStarting the SQL conversion...")
    pipeline = extract_pipeline(pipeline)
    optimizer = True
    query = create_query(pipeline, mlmanager, features, test_table_name, optimizer, debug)
    print("SQL Conversion completed.\n")

    # SQL predict
    print("\nStarting the SQL inference...")
    sql_preds = pd.read_sql(query, conn)
    sql_preds = pd.Series(sql_preds.iloc[:, 0])
    print(sql_preds[:10])
    null_val = False
    if sql_preds.isnull().sum() == 0:
        eval_fn(f"{model_name} SQL", y_test, sql_preds)
    else:
        null_val = True
    print("SQL inference completed.\n")

    # Null value test
    if null_val:
        print("\nNull value test")
        null_val_cnt = 0
        for sample_id in sql_preds[sql_preds.isnull()].index:
            print(sample_id)
            for (attr, val) in zip(test.columns, test.iloc[sample_id, :].values):
                print("\t", attr, '=', val)
            null_val_cnt += 1
        print(f"Found {null_val_cnt} null predictions.")

    # Accuracy test
    print("\nAccuracy test")
    equals = False
    for prec in range(10, 0, -1):
        ml_preds = ml_preds.map(lambda x: round(x, prec))
        sql_preds = sql_preds.map(lambda x: round(x, prec))
        if ml_preds.equals(sql_preds):
            print(f"The prediction scores are equal with {prec} decimal precision.")
            print(":)")
            equals = True
            break
    if not equals:
        print("The prediction scores are not equal.")
        print(":(\n")

        ne_preds = 0
        for i in range(len(ml_preds)):
            ml_pred = ml_preds.iloc[i]
            sql_pred = sql_preds.iloc[i]

            if ml_pred != sql_pred:
                if debug:
                    print(i, ml_pred, sql_pred)
                    for (attr, val) in zip(test.columns, test.iloc[i, :].values):
                        print("\t", attr, '=', val)
                ne_preds += 1
        print(f"Found {ne_preds} incorrect predictions.")

        # if debug:
        #     if 'Tree' in model_name:
        #         sample_id = 0
        #         check_path(fitted_model, test.values, sample_id, plot=False, save_fig=None)


if __name__ == '__main__':

    # BIKE
    # conf = BikeConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler'], 'gbr')

    # CREDITCARD
    # conf = CreditCardConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler'], 'gbc')

    # CRITEO
    # conf = CriteoConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler', 'OneHotEncoder'], 'dtc')

    # FLIGHT
    # conf = FlightConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler', 'OneHotEncoder'], 'dtr')

    # HEART
    # conf = HeartConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler'], 'gbc')

    # IRIS
    # conf = IrisConf()
    # data_conf = conf.get_data_conf()
    # pipeline_conf = conf.get_pipeline(['StandardScaler'], 'gbc')

    # TAXI
    conf = TaxiConf()
    data_conf = conf.get_data_conf()
    pipeline_conf = conf.get_pipeline(['StandardScaler', 'OneHotEncoder'], 'dtr')

    main(data_conf, pipeline_conf, conf.task, debug=False)

