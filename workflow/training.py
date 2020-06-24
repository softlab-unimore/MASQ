from datetime import datetime

from dbconnection.connector import get_table
from mlmodel.mlmanager import MLManager
from mlmodel.loader import save_model

from workflow.utils import get_dataframe, get_dataset, get_train_test
from workflow.scenario import get_train_scenario


def check_train_consistency(scenario):
    if scenario['is_db'] \
            and (scenario['db_url'] is None
                 or scenario['table'] is None):
        raise ValueError('You select DBMS but db_url or table are None')

    if not scenario['is_db'] \
            and scenario['dataset'] is None:
        raise ValueError('You select no DBMS but dataset is empty')

    if not scenario['labels'] or scenario['labels_type'] not in ['table', 'column', 'file']:
        raise ValueError('No correct labels are selected')

    if scenario['labels_type'] == 'table' and not scenario['is_db']:
        raise ValueError('Impossible select table label without DBMS connection')

    if scenario['validation'] > 0 and not scenario['metric']:
        raise ValueError('Impossible execute evaluation without metric')


def main():
    print('Workflow Training')
    t_start = datetime.now()

    scenario = get_train_scenario()
    check_train_consistency(scenario)

    # Dataset
    print('Get Dataset')
    ds = get_dataset(scenario)
    features = ds.columns.to_list()

    # Label
    print('Get Label')

    if scenario['labels_type'] == 'file':
        # Get labels from file
        labels = get_dataframe(scenario['labels'])

        if labels is None:
            raise ValueError('Impossible read label from selected file')

        # Get first column from file
        labels = labels.iloc[:, 0].to_list()

    elif scenario['labels_type'] == 'table':
        # Get labels from table
        labels = get_table(scenario['db_url'], scenario['labels'])
        if ds is None:
            raise ValueError('Impossible read labels from DBMS, please check DBMS input')

        # Get first column from table
        labels = labels.iloc[:, 0].to_list()

    elif scenario['labels_type'] == 'column':
        # Get labels from column
        labels = ds[scenario['labels']].to_list()
        # Remove label column from dataset
        features.remove(scenario['labels'])

    else:
        raise ValueError('Selected the wrong label type')

    # Get train and test from dataset
    x_train, y_train, x_test, y_test = get_train_test(ds[features], labels, scenario['validation'])

    # ML Manager
    print('Create ML Manager')
    manager = MLManager()
    # Set ML model
    manager.select_model(scenario['model'])

    # Set ML transforms
    manager.set_transforms(scenario['transforms'])

    # Training
    print('Training...')
    model = manager.fit(x_train, y_train)

    # Finish training
    t_end = datetime.now()
    print('Execution Time: ', t_end - t_start)

    # Compute evaluation
    if y_test and scenario['metric']:
        y_pred = model.predict(x_test)
        res_evaluation = manager.evaluate(scenario['metric'], y_test, y_pred)
        print('Evaluation: ', res_evaluation)

    print('Save Result Model: train.joblib')
    trained_pipeline_name = '../data/result/train.joblib'
    save_model(model, trained_pipeline_name)
    print(':)')


if __name__ == '__main__':
    main()
