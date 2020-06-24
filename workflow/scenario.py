def get_train_scenario():
    scenario = {
        'is_db': True,
        'dataset': '../data/bike_sharing_no_label.csv',
        'db_url': 'mysql+pymysql://root:password@localhost/MLtoSQL',
        'table': 'bike_sharing_fasttree_no_label',

        'labels_type': 'file',
        'labels': '../data/bike_sharing_label.csv',
        # 'labels_type': 'column',
        # 'labels': 'Label',

        'validation': 10,
        'metric': 'r2_score',

        'model': 'GradientBoostingRegressor',
        'transforms': [
            {
                'transform_type': 'StandardScaler',
                'transform_column': 'Temperature',
            }
        ],
        # 'transforms': [],
    }

    return scenario


def get_test_scenario():
    scenario = {
        'is_db': True,
        'dataset': '../data/bike_sharing_no_label.csv',
        'db_url': 'mysql+pymysql://root:password@localhost/MLtoSQL',
        'table': 'bike_sharing_fasttree_no_label',

        'labels_type': 'file',
        'labels': '../data/bike_sharing_label.csv',
        # 'labels_type': 'column',
        # 'labels': 'Label',

        'metric': 'r2_score',

        'pipeline': '../data/result/train.joblib',
        'run_db': True,

    }

    return scenario


def get_simulation_scenario():
    scenario = {
        'is_db': True,
        'db_url': 'mysql+pymysql://root:password@localhost/MLtoSQL',
        'table': 'bike_sharing_fasttree_no_label',
        'pipeline': '../data/result/train.joblib',
        'batch_size': 64,
        'batch_number': 3,
    }

    return scenario

