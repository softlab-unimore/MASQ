import pandas as pd

from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

from dbconnection.connector import get_connector, \
    get_tables, check_table, check_column, get_column, get_table, execute_query

from mlmodel.mlmanager import MLManager

from mlmodel.loader import save_model


def get_dataframe(data, sep=','):
    try:
        ds = pd.read_csv(data, sep=sep)
    except pd.errors.ParserError:
        ds = None
    return ds


def restart(request):
    request.session.clear()
    return HttpResponseRedirect(reverse('msp:index'))


def index(request):
    if 'step' not in request.session:
        request.session['step'] = 1

    # Step 1: upload file directly
    if request.method == 'POST' and 'uploaded_file' in request.FILES:
        file = request.FILES['uploaded_file']
        print("get file: {}".format(file.name))
        print("type: {}".format(type(file)))
        sep = request.POST['sep'] if request.POST['sep'] else ','
        ds = get_dataframe(file, sep=sep)

        if ds is not None:
            print("dataframe:")
            print(ds.head())

            request.session['dataset_name'] = file.name
            request.session['ds_data'] = ds.to_json()
            request.session['sep'] = sep

            request.session['is_db'] = False

            if 'db_url' in request.session:
                del request.session['db_url']

            request.session['step'] = 2
            return render(request, 'msp/index.html')
        else:
            print('dataframe {} isn\'t valid'.format(file.name))
            return render(request, 'msp/index.html',
                          {'upload_error': 'dataframe {} isn\'t valid'.format(file.name)})

    # Step 1: get database connection
    if request.method == 'POST' and 'db_connection' in request.POST:
        db_connection = request.POST['db_connection']
        print("get connection: {}".format(db_connection))

        engine = get_connector(db_connection)
        if engine is not None:
            request.session['db_url'] = db_connection
            request.session['is_db'] = True

            tables = get_tables(db_connection)
            return render(request, 'msp/index.html', {'tables': tables})
        else:
            return render(request, 'msp/index.html',
                          {'upload_error': 'url {} isn\'t valid'.format(db_connection)})

    # Step 1: get table in DBMS
    if request.method == 'POST' and 'table' in request.POST:
        table = request.POST['table']
        print("get table: {}".format(table))

        request.session['dataset_name'] = table
        request.session['is_db'] = True

        if 'ds_data' in request.session:
            del request.session['ds_data']
            del request.session['sep']

        request.session['step'] = 2
        return render(request, 'msp/index.html')

    # Step 2: select modality
    if request.method == 'POST' and 'mode' in request.POST:
        mode = request.POST['mode']
        print("select modality: {}".format(mode))

        request.session['mode'] = mode

        if mode == 'test':
            request.session['step'] = 3
            return render(request, 'msp/index.html')

        elif mode == 'train':

            # case 1: passing another table inside the DBMS
            if request.session['is_db'] and request.POST['label_table']:
                label_table = request.POST['label_table']
                print('get label table: {}'.format(label_table))

                if not check_table(request.session['db_url'], label_table):
                    return render(request, 'msp/index.html',
                                  {'label_error': 'Table {} doens\t exists'.format(label_table)})
                else:
                    pass

            # case 2
            elif request.POST['label_column']:
                label_column = request.POST['label_column']
                print('get label column: {}'.format(label_column))

                # check column on table in DBMS
                if request.session['is_db']:
                    print('check column in table {}'
                          .format(request.session['dataset_name']))
                    res = check_column(request.session['db_url'],
                                       request.session['dataset_name'],
                                       label_column)
                    if res:
                        labels = get_column(request.session['db_url'],
                                            request.session['dataset_name'],
                                            label_column)
                        print(labels)
                        request.session['labels'] = labels
                        request.session['is_label_column'] = label_column

                        request.session['step'] = 3
                        return render(request, 'msp/index.html')

                # check column on dataframe
                else:
                    print('check column in dataframe {}'.
                          format(request.session['dataset_name']))

                    ds = pd.read_json(request.session['ds_data'])
                    if label_column in ds.columns:
                        labels = ds[label_column].to_list()
                        print(ds[label_column])

                        request.session['labels'] = labels
                        request.session['is_label_column'] = label_column

                        request.session['step'] = 3
                        return render(request, 'msp/index.html')

                return render(request, 'msp/index.html',
                              {'label_error': 'Column {} doesnt exists'.format(label_column)})

            # case 3: upload label file
            elif 'label_dataframe' in request.FILES:
                file = request.FILES['label_dataframe']
                print("get label dataframe: {}".format(file.name))
                print(type(file))

                ds = get_dataframe(file, sep=',')
                print(ds.head())

                if ds is not None:
                    labels = ds.iloc[:, 0].to_list()
                    print(ds.iloc[:, 0])

                    request.session['labels'] = labels
                    request.session['is_label_column'] = False

                    request.session['step'] = 3
                    return render(request, 'msp/index.html')

                return render(request, 'msp/index.html',
                              {'label_error': 'Label Dataframe error'})

        else:
            return render(request, 'msp/index.html',
                          {'label_error': 'Mode {} doesnt exists'.format(mode)})

    # Step 3: add transformations
    if request.method == 'POST' and 'transform_type' in request.POST:
        transform_type = request.POST['transform_type']
        transform_column = request.POST['transform_column']
        error = False

        if request.session['is_db']:
            print('check column {} in table {}'
                  .format(transform_column, request.session['dataset_name']))
            res = check_column(request.session['db_url'],
                               request.session['dataset_name'],
                               transform_column)
            if not res:
                error = True
        else:
            print('check column {} in dataframe'.format(transform_column))
            ds = pd.read_json(request.session['ds_data'])
            if transform_column not in ds.columns:
                error = True

        if error:
            return render(request, 'msp/index.html',
                          {'transform_error': 'Impossible apply transform {} {}'.format(transform_type,
                                                                                        transform_column)})
        else:

            if 'transforms' not in request.session:
                print('create new transform variable')
                request.session['transforms'] = []

            print(request.session['transforms'])
            l = request.session['transforms']
            l.append({
                'transform_type': transform_type,
                'transform_column': transform_column
            })
            request.session['transforms'] = l
            print(request.session['transforms'])

    # Step 3: remove transformations
    if request.method == 'POST' and 'delete_transforms' in request.POST:
        print('delete all transforms')
        if 'transforms' in request.session:
            del request.session['transforms']

    # Step 4: execute
    if request.method == 'POST' and 'model_type' in request.POST:
        print('run model')

        model_type = request.POST['model_type']

        model_file = None
        if request.session['mode'] == 'test':
            if 'uploaded_model' not in request.FILES:
                return render(request, 'msp/index.html',
                              {'model_error': 'No pre trained model uploaded'})

            model_file = request.FILES['uploaded_model']
            # request.session['model_file'] = model_file

        run_db = False
        if request.session['mode'] == 'test' and request.session['is_db']:
            run_db = request.POST['run_db']

        request.session['model_type'] = model_type
        request.session['run_db'] = run_db
        request.session['step'] = 4

        manager = MLManager()
        manager.select_model('GradientBoostingRegressor')

        if request.session['mode'] == 'train':
            if request.session['is_db']:
                ds = get_table(request.session['db_url'], request.session['dataset_name'])
            else:
                ds = pd.read_json(request.session['ds_data'])

            print(ds.head(10))
            print(request.session['is_label_column'])
            if request.session['is_label_column']:
                del ds[request.session['is_label_column']]

            model = manager.fit(ds, request.session['labels'])
            save_model(model, "data/model_{}.joblib".format(model_type))
            print('save_model')

        elif request.session['mode'] == 'test':
            if request.session['is_db']:
                print('runjdfisbgsijkbgjsdlgosdbgoisdhgilsdhgodspngfposifdòlhgaofdis')
                columns = list(get_table(request.session['db_url'], request.session['dataset_name']).columns)
                print(columns)
                query = manager.generate_query(model_file, request.session['dataset_name'], columns)
                print(query)
                ds = execute_query(request.session['db_url'], query)
                print(ds)

            else:
                ds = pd.read_json(request.session['ds_data'])
                y_pred = manager.predict(ds, model_file)
                y_pred = pd.Series(y_pred, )
                y_pred.to_csv("data/{}_prediction.csv".format(model_type), index=False)
                print('save_result')

    return render(request, 'msp/index.html', )
