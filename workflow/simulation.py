from datetime import datetime
import numpy as np
import pandas as pd

from dbconnection.connector import execute_query
from mlmodel.mlmanager import MLManager

from workflow.utils import get_dataset
from workflow.scenario import get_simulation_scenario


def get_batch(ds, i, size):
    return ds.iloc[i * size: (i + 1) * size]


def main():
    print('Workflow Simulation')
    scenario = get_simulation_scenario()

    # Dataset
    print('Get Dataset and Features')
    t_load_start = datetime.now()
    ds = get_dataset(scenario)
    features = ds.columns.to_list()
    t_load_end = datetime.now()

    # ML Manager
    print('Create ML Manager')
    manager = MLManager()

    t_ml = []
    t_db = []

    # Testing Phase
    print('Simulation...')
    for i in range(scenario['batch_number']):
        ds_batch = get_batch(ds, i, scenario['batch_size'])
        if ds_batch.empty:
            break

        # Execute predict using MLManager and ML Library
        print('ML Prediction...')
        t_start = datetime.now()
        _ = manager.predict(ds[features], scenario['pipeline'])
        t_end = datetime.now()

        t_ml.append(t_end - t_start)

        # Create Batch for DBMS
        ds_batch.to_sql('batch', con=scenario["db_url"], if_exists="replace", index=False)

        # Generate query using MLManager
        print('Query Generation...')
        query = manager.generate_query(scenario['pipeline'], scenario['table'], features)

        # Execute query
        print('Query Execution...')
        t_start = datetime.now()
        _ = execute_query(scenario['db_url'], query)
        t_end = datetime.now()

        t_db.append(t_end - t_start)

    # Finish Simulation
    print('ML Execution Time: ', np.mean(t_ml) + (t_load_end - t_load_start))
    print('DB Execution Time: ', np.mean(t_db))

    print(':)')


if __name__ == '__main__':
    main()
