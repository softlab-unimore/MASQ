from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
# from lightning.regression import SDCARegressor
# from sklearn.neural_network import MLPClassifier, MLPRegressor


class Models(object):
    gbc = {
        'name': 'GradientBoostingClassifier',
        'obj': GradientBoostingClassifier(max_leaf_nodes=20, n_estimators=20, min_samples_leaf=10,
                                          learning_rate=0.2, random_state=24),
    }

    gbr = {
        'name': 'GradientBoostingRegressor',
        'obj': GradientBoostingRegressor(max_leaf_nodes=20, n_estimators=20, min_samples_leaf=10,
                                         learning_rate=0.2, random_state=24),
    }

    dtc = {
        'name': 'DecisionTreeClassifier',
        'obj': DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    dtr = {
        'name': 'DecisionTreeRegressor',
        'obj': DecisionTreeRegressor(max_depth=5, random_state=42),
    }

    rfc = {
        'name': 'RandomForestClassifier',
        'obj': RandomForestClassifier(max_depth=5, random_state=24, n_estimators=4),
    }

    rfr = {
        'name': 'RandomForestRegressor',
        'obj': RandomForestRegressor(max_depth=5, random_state=24, n_estimators=2),
    }

    sgdr = {
        'name': 'SGDRegressor',
        'obj': SGDRegressor(random_state=42)
    }

    logr = {
        'name': 'LogisticRegression',
        'obj': LogisticRegression(random_state=24, max_iter=10000)
    }

    linr = {
        'name': 'LinearRegression',
        'obj': LinearRegression()
    }

    # sdcar = {
    #     'name': 'SDCARegressor',
    #     'obj': SDCARegressor()
    # }
    #
    # mlpr = {
    #     'name': 'MLPRegressor',
    #     'obj': MLPRegressor()
    # }
    #
    # mlpc = {
    #     'name': 'MLPClassifier',
    #     'obj': MLPClassifier()
    # }

    models = {
        'gbc': gbc,
        'gbr': gbr,
        'dtc': dtc,
        'dtr': dtr,
        'rfc': rfc,
        'rfr': rfr,
        'sgdr': sgdr,
        'logr': logr,
        'linr': linr,
        # 'sdcar': sdcar,
        # 'mlpr': mlpr,
        # 'mlpc': mlpc,
    }

    regressors = ['gbr', 'dtr', 'rfr', 'sgdr', 'linr']# , 'sdcar', 'mlpr']
    classifiers = ['gbc', 'dtc', 'rfc', 'logr']#, 'mlpc']

    @staticmethod
    def get_model(model_name: str):
        assert isinstance(model_name, str)

        if model_name in Models.models:
            return Models.models[model_name]

        raise ValueError(f"No conf {model_name} found. The conf available are: {list(Models.models)}")


class Transformers(object):
    scaler = {
        'name': 'StandardScaler',
        'obj': StandardScaler(with_mean=False),
        'features': [],
    }

    ohe = {
        'name': 'OneHotEncoder',
        'obj': OneHotEncoder(handle_unknown='ignore'),
        'features': [],
    }

    transformers = {
        'StandardScaler': scaler,
        'OneHotEncoder': ohe,
    }

    @staticmethod
    def get_transformer(transformer_name: str):
        assert isinstance(transformer_name, str)

        if transformer_name in Transformers.transformers:
            return Transformers.transformers[transformer_name]

        raise ValueError(f"No conf {transformer_name} found. The conf available are: {list(Transformers.transformers)}")

    @staticmethod
    def get_transformers(transformers: list):
        assert isinstance(transformers, list)
        assert len(transformers) > 0

        target_transformers = []
        for transformer in transformers:
            target_transformers.append(Transformers.get_transformer(transformer))

        return target_transformers


class BikeConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'bike',
            'train_file': 'bike_train.csv',
            'test_file': 'bike_test.csv',
            'label': 'cnt',
            'train_table_name': 'bike_train',
            'test_table_name': 'bike_test',
        }
        self.task = 'regression'
        self.features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp',
                         'atemp', 'hum', 'windspeed']
        self.cat_features = []
        self.num_features = self.features[:]
        self.available_models = Models.regressors
        self.available_transformers = {'StandardScaler': self.num_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class CreditCardConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'creditcard',
            'train_file': 'creditcard_train.csv',
            'test_file': 'creditcard_test.csv',
            'label': 'Class',
            'train_table_name': 'creditcard_train',
            'test_table_name': 'creditcard_test',
        }
        self.task = 'binary_classification'
        self.features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                         'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                         'Amount']
        self.cat_features = []
        self.num_features = self.features[:]
        self.available_models = Models.classifiers
        self.available_transformers = {'StandardScaler': self.num_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class CriteoConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'criteo',
            'train_file': 'criteo_train.csv',
            'test_file': 'criteo_test.csv',
            'label': 'label',
            'train_table_name': 'criteo_train',
            'test_table_name': 'criteo_test',
        }
        self.task = 'binary_classification'
        self.features = ['feature_integer1', 'feature_integer2', 'feature_integer3', 'feature_integer4',
                         'feature_integer5', 'feature_integer6', 'feature_integer7', 'feature_integer8',
                         'feature_integer9', 'feature_integer10', 'feature_integer11', 'feature_integer12',
                         'feature_integer13', 'categorical_feature1', 'categorical_feature2', 'categorical_feature3',
                         'categorical_feature4', 'categorical_feature5', 'categorical_feature6', 'categorical_feature7',
                         'categorical_feature8', 'categorical_feature9', 'categorical_feature10',
                         'categorical_feature11', 'categorical_feature12', 'categorical_feature13',
                         'categorical_feature14', 'categorical_feature15', 'categorical_feature16',
                         'categorical_feature17', 'categorical_feature18', 'categorical_feature19',
                         'categorical_feature20', 'categorical_feature21', 'categorical_feature22',
                         'categorical_feature23', 'categorical_feature24', 'categorical_feature25',
                         'categorical_feature26']
        self.cat_features = []
        self.num_features = []
        for f in self.features:
            if 'integer' in f:
                self.num_features.append(f)
            if 'categorical' in f:
                self.cat_features.append(f)

        self.available_models = Models.classifiers
        self.available_transformers = {'StandardScaler': self.num_features, 'OneHotEncoder': self.cat_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class FlightConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'flight',
            'train_file': 'flight_train.csv',
            'test_file': 'flight_test.csv',
            'label': 'LateAircraftDelay',
            'train_table_name': 'flight_train',
            'test_table_name': 'flight_test',
        }
        self.task = 'regression'
        self.features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
                         'UniqueCarrier', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay',
                         'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'Diverted',
                         'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay']
        self.cat_features = ["UniqueCarrier", "Origin", "Dest"]
        self.num_features = list(set(self.features).difference(set(self.cat_features)))
        self.available_models = Models.regressors
        self.available_transformers = {'StandardScaler': self.num_features, 'OneHotEncoder': self.cat_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class HeartConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'heart',
            'train_file': 'heart_train.csv',
            'test_file': 'heart_test.csv',
            'label': 'Label',
            'train_table_name': 'heart_train',
            'test_table_name': 'heart_test',
        }
        self.task = 'binary_classification'
        self.features = ["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak",
                         "Slope", "Ca", "Thal"]
        self.cat_features = []
        self.num_features = self.features[:]
        self.available_models = Models.classifiers
        self.available_transformers = {'StandardScaler': self.num_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class IrisConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'iris',
            'train_file': 'iris_train.csv',
            'test_file': 'iris_test.csv',
            'label': '#Label',
            'train_table_name': 'iris_train',
            'test_table_name': 'iris_test',
        }
        self.task = 'multi_classification'
        self.features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        self.cat_features = []
        self.num_features = self.features[:]
        self.available_models = Models.classifiers
        self.available_transformers = {'StandardScaler': self.num_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf


class TaxiConf(object):
    def __init__(self):

        self.data_conf = {
            'use_case': 'taxi',
            'train_file': 'taxi_train.csv',
            'test_file': 'taxi_test.csv',
            'label': 'fare_amount',
            'train_table_name': 'taxi_train',
            'test_table_name': 'taxi_test',
        }
        self.task = 'regression'
        self.features = ['vendor_id', 'rate_code', 'passenger_count', 'trip_time_in_secs', 'trip_distance',
                         'payment_type']
        self.cat_features = ["vendor_id", "rate_code", "payment_type"]
        self.num_features = ["passenger_count", "trip_time_in_secs", "trip_distance"]
        self.available_models = Models.regressors
        self.available_transformers = {'StandardScaler': self.num_features, 'OneHotEncoder': self.cat_features}

    def get_pipeline(self, transformers: list, model: str):
        assert isinstance(transformers, list)
        assert isinstance(model, str)
        assert model in self.available_models, f"No model {model} found for this pipeline."

        if len(transformers) > 0:
            for t in transformers:
                assert isinstance(t, str)
                assert t in self.available_transformers, f"No transformer {t} found for this pipeline."

        out_transformers = []
        if len(transformers) > 0:
            out_transformers = Transformers.get_transformers(transformers)
            for ts in out_transformers:
                ts['features'] = self.available_transformers[ts['name']]
        out_model = Models.get_model(model)

        return {
            'model': out_model,
            'transforms': out_transformers
        }

    def get_data_conf(self):
        return self.data_conf
