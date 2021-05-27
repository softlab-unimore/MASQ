from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
# from lightning.regression import SDCARegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array
from sklearn.tree._tree import DTYPE
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlmodel.loader import load_model
from mlmodel.gradient_boosting.gbm_sql import GBMSQL
from mlmodel.logistic_regression.lr_sql import LogisticRegressionSQL
from mlmodel.standard_normalization.std_scaler_sql import StandardScalerSQL
from mlmodel.sgd_regressor.sgdr_sql import SGDModelSQL
# from mlmodel.sdca_regressor.sdcar_sql import SDCARegressorSQL
from mlmodel.one_hot_encoder.one_hot_encoder_sql import OneHotEncoderSQL
from mlmodel.decision_tree.dtc_sql import DTMSQL
from mlmodel.random_forest.rf_sql import RFMSQL
# from mlmodel.multi_layer_perceptron.mlp_sql import MLPSQL
from mlmodel.linear_regressor.linr_sql import LINRModelSQL

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score

import numpy as np


class MLManager(object):
    model_types = {
        'GradientBoostingRegressor': GradientBoostingRegressor(max_leaf_nodes=20, n_estimators=100, min_samples_leaf=10,
                                                               learning_rate=0.2, random_state=24),
        'GradientBoostingClassifier': GradientBoostingClassifier(max_leaf_nodes=20, n_estimators=100, min_samples_leaf=10,
                                                               learning_rate=0.2, random_state=24),
        'LogisticRegression': LogisticRegression(random_state=24),
        'SGDRegressor': SGDRegressor(),
        # 'SDCARegressor': SDCARegressor(),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=10, random_state=24),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=3, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(max_leaf_nodes=20, n_estimators=100, min_samples_leaf=10, random_state=24),
        'RandomForestRegressor': RandomForestRegressor(max_leaf_nodes=20, n_estimators=100, min_samples_leaf=10, random_state=24),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(5, 5, 5)),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(5, 5, 5)),
        'LinearRegression': LinearRegression(),

    }

    transform_types = {
        'StandardScaler': StandardScaler(with_mean=False),
        'OneHotEncoder': OneHotEncoder(handle_unknown='ignore'),
    }

    sql_model_types = {
        'GradientBoostingRegressor': GBMSQL(),
        'GradientBoostingClassifier': GBMSQL(classification=True),
        'LogisticRegression': LogisticRegressionSQL(),
        'SGDRegressor': SGDModelSQL(),
        # 'SDCARegressor': SDCARegressorSQL(),
        'DecisionTreeClassifier': DTMSQL(classification=True),
        'DecisionTreeRegressor': DTMSQL(),
        'RandomForestClassifier': RFMSQL(classification=True),
        'RandomForestRegressor': RFMSQL(),
        # 'MLPClassifier': MLPSQL(classification=True),
        # 'MLPRegressor': MLPSQL(),
        'LinearRegression': LINRModelSQL()
    }

    sql_transform_types = {
        'StandardScaler': StandardScalerSQL(),
        'OneHotEncoder': OneHotEncoderSQL(),
    }

    metric_types = {
        'r2_score': r2_score,
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score
    }

    model_name = None
    transforms = []

    def select_model(self, model_name):
        if model_name not in self.model_types:
            raise ValueError('{} model isn\'t supported'.format(model_name))
        self.model_name = model_name

    def set_transforms(self, transforms):
        for transform in transforms:
            if 'transform_type' not in transform or 'transform_column' not in transform:
                raise TypeError('transform_column or transform_type aren\'t in transform dictionary')
            if transform['transform_type'] not in self.transform_types:
                raise ValueError('{} transform isn\'t supported'.format(transform.transform_type))

        self.transforms = transforms

    def add_transform(self, transform):
        self.transforms.append(transform)

    def clear_transform(self):
        self.transforms = []

    def _create_pipeline(self):
        transforms = {}
        for transform in self.transforms:
            if transform['transform_type'] not in transforms:
                transforms[transform['transform_type']] = [transform['transform_column']]
            else:
                transforms[transform['transform_type']].append(transform['transform_column'])

        pipeline_transforms = []
        for k, val in transforms.items():
            pipeline_transforms.append(
                (k, self.transform_types[k], val)
            )

        pipeline_transforms = ('pipeline_transforms',
                               ColumnTransformer(remainder='passthrough',
                                                 transformers=pipeline_transforms))

        pipeline_estimator = (self.model_name, self.model_types[self.model_name])

        pipeline = Pipeline(steps=[pipeline_transforms, pipeline_estimator])
        return pipeline

    @staticmethod
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

    @staticmethod
    def extract_pipeline_components(pipeline):
        model_name, _ = pipeline.steps[1]
        transforms = []
        for idx in range(len(pipeline.steps[0][1].transformers)):
            a, _, c = pipeline.steps[0][1].transformers_[idx]

            for col in c:
                transforms.append({
                    'transform_type': a,
                    'transform_column': col,
                })

        return {
            'model': model_name,
            'transforms': transforms
        }

    def _check_fitted_pipeline(self, pipeline, X):

        # Sklearn's GradientBoostingClassifier adds to the final score (i.e., the weighted sum of the tree scores) an
        # init score.
        # This part has not been implemented in SQL, but this score has been added to the final query as an offset.

        # retrieve the Sklearn's GradientBoostingClassifier init score
        if self.model_name.startswith("GradientBoosting"):
            model = pipeline[self.model_name]

            transformers = pipeline["pipeline_transforms"]
            transformed_data = transformers.transform(X)

            transformed_data = check_array(transformed_data, dtype=DTYPE, order="C", accept_sparse='csr')
            init_score = model._raw_predict_init(transformed_data).ravel()[0]
            model.init_score = init_score

    def fit(self, X, y):
        if self.model_name is None:
            raise ValueError('model has not been selected')

        pipeline = self._create_pipeline()

        # if the model comes from the lightning library, y has to be a 2D array
        if self.model_name == "SDCARegressor":
            y = np.array(y).reshape(-1, 1)

        pipeline.fit(X, y)

        self._check_fitted_pipeline(pipeline, X)

        return pipeline

    @staticmethod
    def predict(X, model_data):
        model = load_model(model_data)
        y_pred = model.predict(X)
        return y_pred

    def generate_query(self, model_data, dataset_name, features, optimization):
        model = load_model(model_data)
        pipeline = self.extract_pipeline(model)
        input_table = dataset_name

        if 'index' in features:
            features.remove('index')

        opt = Optimizer(pipeline, features, optimization)
        pipeline = opt.optimize()

        # create an SQL query for each transformer
        sql_transformers = pipeline["transforms"]
        for sql_transformer in sql_transformers:
            transformation_query = sql_transformer.query(input_table)

            # the input table for the possible next transformer is the output of the current transformer
            input_table = "({}) AS data".format(transformation_query)

        # create the SQL query that performs the model inference
        model = pipeline['model']
        fitted_model = model["trained_model"]
        model_sql_wrapper = model['model_sql_wrapper']
        assert 'model_features' in model
        if model['model_features'] is not None:
            features = model['model_features']

        query = model_sql_wrapper.query(fitted_model, features, input_table)

        return query

    def evaluate(self, metric, labels, predictions):
        if metric not in self.metric_types:
            raise ValueError('metric {} is not defined'.format(metric))

        res = self.metric_types[metric](labels, predictions)
        return res


class Optimizer(object):
    def __init__(self, pipeline: dict, features: list, optimization):
        assert isinstance(pipeline, dict)
        assert 'model' in pipeline
        assert 'transforms' in pipeline
        assert 'model_name' in pipeline['model']
        assert 'trained_model' in pipeline['model']
        assert isinstance(pipeline['transforms'], list)
        assert isinstance(features, list)
        assert len(features) > 0
        for f in features:
            assert isinstance(f, str)
        assert isinstance(optimization, bool)

        self.pipeline = pipeline
        self.features = features

        self.transformers = self.pipeline["transforms"]
        self.model = self.pipeline['model']
        self.transformer_names = [transformer["transform_name"] for transformer in self.transformers]
        self.model_name = self.model["model_name"]
        self.tree_based_model_keys = ['Gradient', 'Tree', 'Forest']

        self.ml_manager = MLManager()

        if optimization:
            self.operator_fusion = True
            print("Optimizer enabled.")
        else:
            self.operator_fusion = False
            print("Optimizer disabled.")

    def optimize(self):

        out_pipeline = self.pipeline.copy()
        features = self.features

        transformers_to_merge = []

        if self.operator_fusion:

            # if the pipeline includes an OneHotEncoder and a tree-based model then apply the one-hot encoding directly
            # in the decision tree rules
            if 'OneHotEncoder' in self.transformer_names and \
                    any(key in self.model_name for key in self.tree_based_model_keys):
                transformers_to_merge.append('OneHotEncoder')

        # get the fitted transformers from the pipeline
        prev_transform_features = []
        new_transformers = []
        sql_transformers_to_merge = []
        for transformer in self.transformers:
            transformer_name = transformer["transform_name"]
            fitted_transformer = transformer["fitted_transform"]
            transformer_features = transformer["transform_features"]

            # retrieve the SQL wrapper related to the current transformer and extract its fitted params
            transformer_sql_wrapper = self.ml_manager.sql_transform_types[transformer_name]
            transformer_params = transformer_sql_wrapper.get_params(fitted_transformer, transformer_features,
                                                                    features, prev_transform_features)
            features = transformer_params["out_all_features"]

            # transformers that have to be merged with the model are ignored in this phase
            if transformer_name not in transformers_to_merge:
                new_transformers.append(transformer_sql_wrapper)
            else:
                sql_transformers_to_merge.append((transformer_name, transformer_sql_wrapper, transformer_params))

            prev_transform_features = transformer_params["out_transform_features"][:]

        # get the fitted model from the pipeline and retrieve its SQL wrapper
        fitted_model = self.model["trained_model"]
        model_sql_wrapper = self.ml_manager.sql_model_types[self.model_name]

        if not self.operator_fusion:
            assert len(sql_transformers_to_merge) == 0
            model_sql_wrapper.reset_optimization()

        # merge the previously selected transformers with the model
        for sql_transf_to_merge in sql_transformers_to_merge:
            transf_name = sql_transf_to_merge[0]
            transf_sql = sql_transf_to_merge[1]
            transf_params = sql_transf_to_merge[2]

            if transf_name == 'OneHotEncoder':
                model_sql_wrapper.merge_ohe_with_trees(transf_params['ohe_encoding'])

        new_model = {
            'trained_model': fitted_model,
            'model_sql_wrapper': model_sql_wrapper,
            'model_features': features,
        }

        out_pipeline['transforms'] = new_transformers
        out_pipeline['model'] = new_model

        return out_pipeline
