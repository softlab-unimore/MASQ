from django.http import HttpResponse, FileResponse, Http404

from django.core.files import File

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser, MultiPartParser, FormParser

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

from msp.api.serializers import ScenarioSerializer, ResultScenarioSerializer

from mlmodel.mlmanager import MLManager
from mlmodel.loader import load_model, save_model
from dbconnection.connector import get_tables, get_columns, \
    get_table, get_column, execute_query
from msp.models import Scenario, ResultScenario, Document

from workflow.training import get_train_test


def get_scenario_object(scenario_id):
    """Get Result Scenario from scenario id"""
    try:
        if Scenario.objects.filter(id=scenario_id):
            return Scenario.objects.filter(id=scenario_id)[0]
        else:
            return None
    except ResultScenario.DoesNotExist:
        return None


def get_result_scenario_object(scenario_id):
    """Get Result Scenario from scenario id"""
    try:
        if ResultScenario.objects.filter(scenario__id=scenario_id):
            return ResultScenario.objects.filter(scenario__id=scenario_id)[0]
        else:
            return None
    except ResultScenario.DoesNotExist:
        return None


def get_document_object(filename):
    """Get Document Object with a given filename"""
    try:
        if Document.objects.filter(filename=filename):
            return Document.objects.filter(filename=filename)[0]
        else:
            return None
    except Document.DoesNotExist:
        return None


def get_dataframe(file):
    try:
        ds = pd.read_csv(file.path, sep=',')
    except (pd.errors.ParserError, ValueError):
        return None

    return ds


def get_dataset(scenario):
    if scenario.is_db:
        ds = get_table(scenario.db_url, scenario.table)
    else:
        ds = get_dataframe(scenario.dataset.file)
    return ds


class DocumentDetail(APIView):
    """Manage Document file retrieve from filename"""

    parser_classes = [MultiPartParser, FormParser]

    def get(self, request, filename):
        # Get request type key
        res_type = self.request.query_params.get('type', None)

        # Retrieve document object from filename
        document = get_document_object(filename)
        if not document:
            raise Http404

        if not res_type:
            # The document file is created and passed to the client
            response = FileResponse(document.file, content_type='application/octet-stream')
            response['Content-Length'] = document.file.size
            response['Content-Disposition'] = 'attachment; filename="%s"' % document.filename
            return response

        elif res_type == 'columns':
            # The dataset column are retrieved and parsed
            ds = get_dataframe(document.file)
            if ds is None:
                # Error during dataset parsing, then document isn't a dataframe
                return Response({'detail': 'The selected file is not a dataframe'},
                                status=status.HTTP_400_BAD_REQUEST)

            return Response({'columns': ds.columns.to_list()},
                            status=status.HTTP_200_OK)
        else:
            raise Http404

    def post(self, request, filename):
        # Retrieve file from request object
        file = request.data['file']

        # Retrieve document object from filename
        document = get_document_object(filename)
        s = status.HTTP_200_OK
        if not document:
            # The document object doesn't exists and we create a new one
            document = Document(
                filename=filename,
                file=file,
            )
            document.save()
            s = status.HTTP_201_CREATED

        return Response({'filename': filename}, status=s)


class MLManagerList(APIView):
    """
    List all available models and transforms in ML Manager
    """

    def get(self, request):
        # Get available models and transforms
        # Get type of request from query params
        res_type = self.request.query_params.get('type', None)

        # List of model types
        model_types = list(MLManager.model_types.keys())
        # List of transform types
        transform_types = list(MLManager.transform_types.keys())
        # List of metric types
        metric_types = list(MLManager.metric_types.keys())

        res = {}
        if res_type == 'model':
            res['model_types'] = model_types
        elif res_type == 'transform':
            res['transform_types'] = transform_types
        elif res_type == 'metric':
            res['metric_types'] = metric_types
        else:
            res['model_types'] = model_types
            res['transform_types'] = transform_types
            res['metric_types'] = metric_types

        return Response(res)


class DBMSDetail(APIView):
    """Manage and Check DBMS connection"""

    def get(self, request):
        """Check DBMS connection and get available tables or columns of selected table"""
        # Get dbms_url for query params
        dbms_url = self.request.query_params.get('dbms_url', None)

        # Get table from query params
        table = self.request.query_params.get('table', None)

        print(dbms_url)
        if dbms_url:
            # Retrieve tables from dbms
            tables = get_tables(dbms_url)
            if not tables:
                raise Http404
            if table:
                # Get columns from selected table
                columns = get_columns(dbms_url, table)
                if not columns:
                    raise Http404
                else:
                    return Response({'columns': columns}, status=status.HTTP_200_OK)
            else:
                return Response({'tables': tables}, status=status.HTTP_200_OK)

        raise Http404


class PipelineDetail(APIView):
    """Retrieve pipeline transformation and models"""

    def get(self, request, filename):
        # Retrieve pipeline document object
        document = get_document_object(filename)
        if not document:
            raise Http404

        # Load pipeline model from pipeline file
        pipeline = load_model(document.file)
        if not pipeline:
            return Response({'detail': 'The selected file isn\'t a pipeline objects'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Extract pipeline information from loaded model
        pipeline = MLManager.extract_pipeline_components(pipeline)
        if not pipeline:
            return Response({'detail': 'The selected file isn\'t a pipeline objects'},
                            status=status.HTTP_400_BAD_REQUEST)

        return Response(pipeline, status=status.HTTP_200_OK)


class ScenarioList(APIView):
    """
    List all executed scenarios
    """

    def get(self, request):
        scenarios = Scenario.objects.all()
        serializer = ScenarioSerializer(scenarios, many=True)
        return Response(serializer.data[::-1])


class ResultScenarioDetail(APIView):
    """Get Result Scenario Object from a given Scenario Id"""

    def get(self, request):
        # Get scenario id from query params
        scenario_id = self.request.query_params.get('id', None)
        # Get Result Scenario from scenario id
        result_scenario = get_result_scenario_object(scenario_id)
        if not result_scenario:
            raise Http404
        # Serialize Result Scenario object
        serializer = ResultScenarioSerializer(result_scenario)
        return Response(serializer.data)


class ScenarioCompleteDetail(APIView):
    """Get Complete Scenario with Scenario and Result objects"""

    def get(self, request):
        # Get scenario id from query params
        scenario_id = self.request.query_params.get('id', None)

        # Get Result Scenario from scenario id
        result_scenario = get_result_scenario_object(scenario_id)
        if not result_scenario:
            raise Http404

        # Serialize result object
        result_serializer = ResultScenarioSerializer(result_scenario)
        result_serializer = result_serializer.data

        # Serialize scenario object
        scenario_serializer = ScenarioSerializer(result_scenario.scenario)
        scenario_serializer = scenario_serializer.data

        # Create complete scenario object
        result_serializer.update(scenario_serializer)
        return Response(result_serializer)


class ScenarioDetail(APIView):
    """Manage Scenario object with retrieve, delete and upload operation"""

    def get(self, request):
        # Retrieve scenario object
        scenario_id = self.request.query_params.get('id', None)
        scenario = get_scenario_object(scenario_id)
        if not scenario:
            raise Http404

        serializer = ScenarioSerializer(scenario)
        return Response(serializer.data)

    def delete(self, request):
        # Delete scenario object
        scenario_id = self.request.query_params.get('id', None)

        # Retrieve result scenario object
        result_scenario = get_result_scenario_object(scenario_id)
        if not result_scenario:
            raise Http404

        # Retrieve scenario object from result one
        scenario = result_scenario.scenario
        if not scenario:
            raise Http404

        # Retrieve result file
        result_file = Document.objects.filter(filename=result_scenario.file_result)
        if not result_file:
            print('The result scenario file doesn\'t exists')
            # return Response({'detail': 'The result scenario file doesn\'t exists'},
            #                 status=status.HTTP_400_BAD_REQUEST)

        else:
            result_file = result_file[0]

            # Check if the result pipeline is used in different scenario
            query = Scenario.objects.filter(pipeline__filename=result_scenario.file_result)
            if not query:
                # Delete file
                result_file.file.delete()
                # Delete document object
                result_file.delete()

        # Delete result scenario object
        result_scenario.delete()

        # Delete dataset object if it's never used
        if scenario.dataset:
            query = Scenario.objects.filter(dataset=scenario.dataset)
            if len(query) < 2:
                # Delete dataset file
                scenario.dataset.file.delete()
                # Delete dataset object
                scenario.dataset.delete()

        # Delete pipeline object if it's never used
        if scenario.pipeline:
            # Delete pipeline only is never used in result scenario or other scenarios
            query1 = Scenario.objects.filter(pipeline=scenario.pipeline)
            query2 = ResultScenario.objects.filter(file_result=scenario.pipeline.filename)
            if len(query1) + len(query2) < 2:
                scenario.pipeline.file.delete()
                scenario.pipeline.delete()

        # Delete labels file if it exists and it's never used
        if scenario.labels_type == 'file':
            query = Scenario.objects.filter(labels=scenario.labels)
            if len(query) < 2:
                document = get_document_object(scenario.labels)
                if document:
                    document.file.delete()
                    document.delete()

        # Delete scenario
        scenario.delete()
        return Response(status=status.HTTP_200_OK)


def create_train_scenario(data):
    scenario = Scenario()

    # Dataset params
    scenario.is_db = data.get('is_db')

    # DBMS url and table
    scenario.db_url = data.get('db_url')
    scenario.table = data.get('table')

    # Dataset filename to Document
    scenario.dataset = get_document_object(data.get('dataset'))

    # Modality params
    scenario.mode = 'train'

    # Get labels
    scenario.labels_type = data.get('labels_type')
    scenario.labels = data.get('labels')

    # Get validation option
    scenario.validation = data.get('validation', 0)
    scenario.metric = data.get('metric')

    # Model params
    scenario.model = data.get('model')
    scenario.transforms = json.dumps(data.get('transforms', []))

    return scenario


def check_train_consistency(scenario):
    if scenario.is_db \
            and (scenario.db_url is None
                 or scenario.table is None):
        raise ValueError('You select DBMS but db_url or table are None')

    if not scenario.is_db \
            and scenario.dataset is None:
        raise ValueError('You select no DBMS but dataset is empty')

    if not scenario.labels or scenario.labels_type not in ['table', 'column', 'file']:
        raise ValueError('No correct labels are selected')

    if scenario.labels_type == 'table' and not scenario.is_db:
        raise ValueError('Impossible select table label without DBMS connection')

    if scenario.validation > 0 and not scenario.metric:
        raise ValueError('Impossible execute evaluation without metric')


class ScenarioTrainDetail(APIView):

    def post(self, request):
        t_start = datetime.now()
        scenario: Scenario = create_train_scenario(request.data)

        try:
            check_train_consistency(scenario)
        except ValueError as e:
            return Response({
                'detail': str(e),
            }, status=status.HTTP_400_BAD_REQUEST)

        # Dataset
        print('Get Dataset')
        ds = get_dataset(scenario)
        if ds is None:
            return Response({
                'detail': 'Impossible read the selected dataset',
            }, status=status.HTTP_400_BAD_REQUEST)

        features = ds.columns.to_list()

        # Label
        print('Get Label')

        if scenario.labels_type == 'file':
            # Get labels from file
            labels_document = get_document_object(scenario.labels)
            labels = get_dataframe(labels_document.file)

            if labels is None:
                return Response({'detail': 'The selected labels file {} isn\'t valid'
                                .format(labels_document.filename)},
                                status=status.HTTP_400_BAD_REQUEST)

            # Get first column from file
            labels = labels.iloc[:, 0].to_list()

        elif scenario.labels_type == 'table':
            # Get labels from table
            labels = get_table(scenario.db_url, scenario.labels)
            if not labels:
                return Response({'detail': 'The selected table {} isn\'t valid for label'
                                .format(scenario.labels)},
                                status=status.HTTP_400_BAD_REQUEST)

            # Get first column from table
            labels = labels.iloc[:, 0].to_list()

        elif scenario.labels_type == 'column':
            # Get labels from column
            if scenario.labels not in ds.columns:
                return Response({'detail': 'The selected column {} isn\'t valid for label'
                                .format(scenario.labels)},
                                status=status.HTTP_400_BAD_REQUEST)

            labels = ds[scenario.labels].to_list()
            # Remove label column from dataset
            features.remove(scenario.labels)

        else:
            return Response({'detail': 'Select the wrong labels type'}, status=status.HTTP_400_BAD_REQUEST)

        # Get train and test from dataset
        x_train, y_train, x_test, y_test = get_train_test(ds[features], labels, scenario.validation)

        # ML Manager
        manager = MLManager()
        # Set ML model
        manager.select_model(scenario.model)

        # Set ML transforms
        manager.set_transforms(json.loads(scenario.transforms))

        # Training
        model = manager.fit(x_train, y_train)

        # Finish training
        t_end = datetime.now()

        # Compute evaluation
        res_evaluation = None
        if y_test and scenario.metric:
            y_pred = model.predict(x_test)
            res_evaluation = manager.evaluate(scenario.metric, y_test, y_pred)

        # Create joblib file
        trained_pipeline_name = "train_{}_{}.joblib".format(scenario.model, datetime.now())
        trained_pipeline_name = trained_pipeline_name.replace(' ', '_')
        trained_pipeline_name = trained_pipeline_name.replace(':', '_')
        save_model(model, trained_pipeline_name)

        # Save trained pipeline in Document model
        f = open(trained_pipeline_name, 'rb')
        document = Document(file=File(f), filename=trained_pipeline_name)
        document.save()
        f.close()

        # Remove joblib file
        os.remove(trained_pipeline_name)

        # Save Scenario model
        scenario.save()

        # Save ResultScenario
        result_scenario = ResultScenario()
        result_scenario.scenario = scenario
        result_scenario.execution_time = (t_end - t_start).total_seconds()
        result_scenario.throughput = result_scenario.execution_time / len(x_train)
        result_scenario.score = res_evaluation
        result_scenario.file_result = document.filename
        result_scenario.save()

        return Response({'detail': 'Pipeline trained correctly',
                         'filename': trained_pipeline_name,
                         'scenario_id': scenario.id},
                        status=status.HTTP_201_CREATED)


def create_test_scenario(data):
    scenario = Scenario()

    # Dataset params
    scenario.is_db = data.get('is_db')

    # DBMS url and table
    scenario.db_url = data.get('db_url')
    scenario.table = data.get('table')

    # Dataset filename to Document
    scenario.dataset = get_document_object(data.get('dataset'))

    # Modality params
    scenario.mode = 'test'

    # Get labels
    scenario.labels_type = data.get('labels_type')
    scenario.labels = data.get('labels')

    # Get validation option
    scenario.metric = data.get('metric')

    # Model params
    scenario.pipeline = get_document_object(data.get('pipeline'))
    scenario.run_db = data.get('run_db')
    scenario.optimizer = data.get('optimizer', False)

    return scenario


def check_test_consistency(scenario):
    if scenario.is_db \
            and (scenario.db_url is None
                 or scenario.table is None):
        raise ValueError('You select DBMS but db_url or table are None')

    if not scenario.is_db \
            and scenario.dataset is None:
        raise ValueError('You select no DBMS but dataset is empty')

    if (not scenario.labels or not scenario.metric) \
            and scenario.labels_type in ['table', 'column', 'file']:
        raise ValueError('No correct labels are selected')

    if scenario.labels_type not in ['table', 'column', 'file', None]:
        raise ValueError('No correct label types are selected')

    if scenario.labels_type == 'table' and not scenario.is_db:
        raise ValueError('Impossible select table label without DBMS connection')

    if not scenario.is_db and scenario.run_db:
        raise ValueError('Impossible create query on db without DBMS connection')

    if not scenario.pipeline:
        raise ValueError('No pipeline is selected')


class ScenarioTestDetail(APIView):

    def post(self, request):

        scenario = create_test_scenario(request.data)
        inference_time = 0

        try:
            check_test_consistency(scenario)
        except ValueError as e:
            return Response({
                'detail': str(e),
            }, status=status.HTTP_400_BAD_REQUEST)

        # Load pipeline model from pipeline file
        pipeline = load_model(scenario.pipeline.file)
        if not pipeline:
            return Response({'detail': 'The selected file isn\'t a pipeline objects'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Extract pipeline information from loaded model
        pipeline = MLManager.extract_pipeline_components(pipeline)
        if not pipeline:
            return Response({'detail': 'The selected file isn\'t a pipeline objects'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Model params
        scenario.model = pipeline.get('model')
        scenario.transforms = json.dumps(pipeline.get('transforms', []))

        # Dataset
        if scenario.run_db:
            # Get features from table
            features = get_columns(scenario.db_url, scenario.table)

        else:
            data_extractor_start = datetime.now()
            # Get Dataset
            ds = get_dataset(scenario)
            features = ds.columns.to_list()

            data_extractor_end = datetime.now()
            data_extractor_time = (data_extractor_end - data_extractor_start).total_seconds()
            inference_time += data_extractor_time

        if scenario.labels_type == 'column':
            # Remove Label column if exists
            features.remove(scenario.labels)

        # ML Manager
        manager = MLManager()

        # Testing Phase
        query = None
        if scenario.run_db:

            inference_start = datetime.now()

            # Generate query using MLManager
            query = manager.generate_query(scenario.pipeline.file, scenario.table, features, scenario.optimizer)

            # Execute query
            y_pred = execute_query(scenario.db_url, query)
            y_pred = pd.Series(y_pred.iloc[:, 0], name='Label')

            inference_end = datetime.now()
            inference_time += (inference_end - inference_start).total_seconds()

        else:

            inference_start = datetime.now()

            # Execute predict using MLManager and ML Library
            y_pred = manager.predict(ds[features], scenario.pipeline.file)
            y_pred = pd.Series(y_pred, name='Label')

            inference_end = datetime.now()
            inference_time += (inference_end - inference_start).total_seconds()

        # Label
        labels = []
        # Compute evaluation
        if scenario.labels_type:
            if scenario.labels_type == 'file':
                # Get labels from file
                labels_document = get_document_object(scenario.labels)
                labels = get_dataframe(labels_document.file)

                if labels is None:
                    return Response({'detail': 'The selected labels file {} isn\'t valid'
                                    .format(labels_document.filename)},
                                    status=status.HTTP_400_BAD_REQUEST)

                # Get first column from file
                labels = labels.iloc[:, 0].to_list()

            elif scenario.labels_type == 'table':
                # Get labels from table
                labels = get_table(scenario.db_url, scenario.labels)
                if not labels:
                    return Response({'detail': 'The selected table {} isn\'t valid for label'
                                    .format(scenario.labels)},
                                    status=status.HTTP_400_BAD_REQUEST)

                # Get first column from table
                labels = labels.iloc[:, 0].to_list()

            elif scenario.labels_type == 'column' and not scenario.run_db:
                # Get labels from column
                labels = ds[scenario.labels].to_list()

            elif scenario.labels_type == 'column' and scenario.run_db:
                # Get labels from table
                labels = get_column(scenario.db_url, scenario.table, scenario.labels)
            else:
                return Response({'detail': 'Select the wrong labels type'}, status=status.HTTP_400_BAD_REQUEST)

        # Compute evaluation
        res_evaluation = None
        if labels and scenario.metric:
            res_evaluation = manager.evaluate(scenario.metric, labels, y_pred)

        # Create predictions file
        test_result_name = "test_{}_{}.csv".format(scenario.model, datetime.now())
        test_result_name = test_result_name.replace(' ', '_')
        test_result_name = test_result_name.replace(':', '_')
        y_pred.to_csv(test_result_name, index=False, header=True)

        # Save predictions in Document model
        f = open(test_result_name, 'rb')
        document = Document(file=File(f), filename=test_result_name)
        document.save()
        f.close()

        # Remove temporally predictions file
        os.remove(test_result_name)

        # Save Scenario model
        scenario.save()

        # Save ResultScenario
        result_scenario = ResultScenario()
        result_scenario.scenario = scenario
        result_scenario.execution_time = inference_time
        result_scenario.throughput = result_scenario.execution_time / len(y_pred)
        result_scenario.score = res_evaluation
        result_scenario.file_result = document.filename
        result_scenario.query = query
        result_scenario.save()

        return Response({'detail': 'Successfully predicted result',
                         'filename': test_result_name,
                         'scenario_id': scenario.id},
                        status=status.HTTP_201_CREATED)


def check_simulation_consistency(scenario):
    if not scenario['is_db'] or scenario['db_url'] is None or scenario['table'] is None:
        raise ValueError('You select DBMS but db_url or table are None')

    if not scenario['pipeline']:
        raise ValueError('No pipeline is selected')


def create_simulation_scenario(data):
    scenario = {
        'is_db': True,
        'db_url': data.get('db_url'),
        'table': data.get('table'),
        'labels_type': data.get('labels_type'),
        'labels': data.get('labels'),
        'pipeline': get_document_object(data.get('pipeline')),
        'batch_size': data.get('batch_size', 64),
        'batch_number': data.get('batch_number', 1),
        'optimizer': data.get('optimizer', False),
    }
    return scenario


def get_batch(ds, i, size):
    return ds.iloc[i * size: (i + 1) * size]


class ScenarioFastDetail(APIView):

    def post(self, request):
        # Params
        scenario = create_simulation_scenario(request.data)

        try:
            check_simulation_consistency(scenario)
        except ValueError as e:
            return Response({
                'detail': str(e),
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get Dataset
        t_load_start = datetime.now()
        ds = get_table(scenario['db_url'], scenario['table'])
        features = ds.columns.to_list()

        if scenario['labels_type'] == 'column':
            # Remove Label column if exists
            features.remove(scenario['labels'])

        t_load_end = datetime.now()

        # ML Manager
        manager = MLManager()

        t_ml = []
        t_db = []

        # Testing Phase
        for i in range(scenario['batch_number']):
            ds_batch = get_batch(ds, i, scenario['batch_size'])
            if ds_batch.empty:
                break

            # Execute predict using MLManager and ML Library
            t_start = datetime.now()
            _ = manager.predict(ds[features], scenario['pipeline'].file)
            t_end = datetime.now()

            t_ml.append(t_end - t_start)

            # Create Batch for DBMS
            ds_batch.to_sql('batch', con=scenario["db_url"], if_exists="replace", index=False)

            # Generate query using MLManager
            query = manager.generate_query(scenario['pipeline'].file, scenario['table'], features, scenario['optimizer'])

            # Execute query
            t_start = datetime.now()
            _ = execute_query(scenario['db_url'], query)
            t_end = datetime.now()

            t_db.append(t_end - t_start)

        # Finish Simulation
        return Response({'detail': 'Successfully predicted result',
                         'ml_results': {
                             'execution_time': (np.mean(t_ml) + (t_load_end - t_load_start))
                         },
                         'dbms_results': {
                             'execution_time': np.mean(t_db)
                         },
                         },
                        status=status.HTTP_200_OK)
