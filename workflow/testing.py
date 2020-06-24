from datetime import datetime
import pandas as pd

from dbconnection.connector import execute_query, get_table, get_columns, get_column
from mlmodel.mlmanager import MLManager

from workflow.utils import get_dataframe, get_dataset
from workflow.scenario import get_test_scenario


def check_test_consistency(scenario):
    if scenario['is_db'] \
            and (scenario['db_url'] is None
                 or scenario['table'] is None):
        raise ValueError('You select DBMS but db_url or table are None')

    if not scenario['is_db'] \
            and scenario['dataset'] is None:
        raise ValueError('You select no DBMS but dataset is empty')

    if (not scenario['labels'] or not scenario['metric']) \
            and scenario['labels_type'] in ['table', 'column', 'file']:
        raise ValueError('No correct labels are selected')

    if scenario['labels_type'] not in ['table', 'column', 'file', None]:
        raise ValueError('No correct label types are selected')

    if scenario['labels_type'] == 'table' and not scenario['is_db']:
        raise ValueError('Impossible select table label without DBMS connection')

    if not scenario['is_db'] and scenario['run_db']:
        raise ValueError('Impossible create query on db without DBMS connection')

    if not scenario['pipeline']:
        raise ValueError('No pipeline is selected')


def main():
    print('Workflow Testing')
    t_start = datetime.now()

    scenario = get_test_scenario()
    check_test_consistency(scenario)

    # Dataset
    if scenario['run_db']:
        # Get features from table
        print('Extract Features')
        features = get_columns(scenario['db_url'], scenario['table'])
    else:
        # Get Dataset
        print('Get Dataset and Features')
        ds = get_dataset(scenario)
        features = ds.columns.to_list()

    if scenario['labels_type'] == 'column':
        # Remove Label column if exists
        features.remove(scenario['labels'])

    # ML Manager
    print('Create ML Manager')
    manager = MLManager()

    # Testing Phase
    print('Testing...')

    if scenario['run_db']:
        # Generate query using MLManager
        print('Query Generation...')
        query = manager.generate_query(scenario['pipeline'], scenario['table'], features)

        # Execute query
        print('Query Execution...')
        y_pred = execute_query(scenario['db_url'], query)
        y_pred = pd.Series(y_pred.iloc[:, 0], name='Label')

    else:
        # Execute predict using MLManager and ML Library
        y_pred = manager.predict(ds[features], scenario['pipeline'])
        y_pred = pd.Series(y_pred.flatten(), name='Label')

    # Finish testing
    t_end = datetime.now()
    print('Execution Time: ', t_end - t_start)

    print('Save prediction: test.csv')
    # test_result_name = '../data/result/test_run_db{}.csv'.format(scenario['run_db'])
    test_result_name = '../data/result/test.csv'
    y_pred.to_csv(test_result_name, index=False, header=True)

    # Compute evaluation
    if scenario['labels_type']:

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

        elif scenario['labels_type'] == 'column' and not scenario['is_db']:
            # Get labels from column
            labels = ds[scenario['labels']].to_list()

        elif scenario['labels_type'] == 'column' and scenario['is_db']:
            # Get labels from table
            labels = get_column(scenario['db_url'], scenario['table'], scenario['labels'])

        else:
            raise ValueError('Select the wrong label type')

        res_evaluation = manager.evaluate(scenario['metric'], labels, y_pred)
        print('Evaluation: ', res_evaluation)

    print(':)')


if __name__ == '__main__':
    main()
