from sklearn.linear_model import SGDRegressor, SGDClassifier
from collections import Iterable
from utils.dbms_utils import DBMSUtils


class SGDModelSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn's SGDRegressor/Classifier object.
    """

    def __init__(self):
        self.dbms = None
        self.mode = None

    def set_mode(self, mode: str):
        assert isinstance(mode, str), "Wrong data type for param 'mode'."
        self.mode = mode

    def set_dbms(self, dbms: str):
        self.dbms = dbms

    def reset_optimization(self):
        pass

    @staticmethod
    def get_params(sgd_model):
        """
        This method extracts from the SGDRegressor/Classifer the fitted parameters (i.e., weights and intercept).

        :param sgd_model: the fitted Sklearn SGDRegressor/Classifier object
        :return: Python dictionary containing the fitted parameters extracted from the Sklearn's SGDRegressor/Classifier
        """

        if not isinstance(sgd_model, (SGDRegressor, SGDClassifier)):
            raise TypeError(
                "Wrong data type for parameter sgd_model. Only Sklearn SGDRegressor/Classifier data type is allowed.")

        weights = sgd_model.coef_.ravel()
        bias = sgd_model.intercept_[0]

        return {"weights": weights, "bias": bias}

    @staticmethod
    def _sql_regression_part1(weights, columns, table_name, dbms: str):
        """
        This method creates the portion of the SQL query responsible for the application of the dot product between
        regression weights and features.

        :param weights: the regression weights
        :param columns: the feature names
        :param table_name: the name of the table or the subquery where to read the data
        :param dbms: the name of the dbms
        :return: the portion of the SQL query which implements the regression dot products
        """

        if not isinstance(weights, Iterable):
            raise TypeError("Wrong data type for parameter weights. Only iterable data type is allowed.")

        if not isinstance(columns, Iterable):
            raise TypeError("Wrong data type for parameter columns. Only iterable data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        for weight in weights:
            if not isinstance(weight, float):
                raise TypeError("Wrong data type for weights elements. Only float data type is allowed.")

        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Wrong data type for columns elements. Only string data type is allowed.")

        dbms_util = DBMSUtils()

        query = "SELECT "
        for i in range(len(columns)):
            col = dbms_util.get_delimited_col(dbms, columns[i])
            query += "({} * {}) AS {} ,".format(col, weights[i], col)

        query = query[:-1]  # remove the last ','

        query += " FROM {}".format(table_name)

        return query

    @staticmethod
    def _sql_regression_part2(bias, columns, table_name, dbms: str):
        """
        This method creates the portion of the SQL query responsible for the application of the linear combination over
        the regression dot products.

        :param bias: the regression bias
        :param columns: the feature names
        :param table_name: the name of the table or the subquery where to read the data
        :param dbms: the name of the dbms
        :return: the portion of the SQL query which implements the regression dot product linear combination
        """

        if not isinstance(bias, float):
            raise TypeError("Wrong data type for parameter bias. Only float data type is allowed.")

        if not isinstance(columns, Iterable):
            raise TypeError("Wrong data type for parameter columns. Only iterable data type is allowed.")

        if table_name is not None:
            if not isinstance(table_name, str):
                raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Wrong data type for columns elements. Only string data type is allowed.")

        dbms_util = DBMSUtils()

        query = "SELECT "
        query += " ( "
        for col in columns:
            query += "{} +".format(dbms_util.get_delimited_col(dbms, col))

        query += "{}".format(bias)
        query += ") AS Score"

        query += " FROM {}".format(table_name)

        return query

    def query(self, sgd_model, features, table_name):
        """
        This method creates the SQL query that performs the SGDRegressor/Classifier inference.

        :param sgd_model: the fitted Sklearn SGDRegressor/Classifier object
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the SGDRegressor/Classifier inference
        """

        if not isinstance(sgd_model, (SGDRegressor, SGDClassifier)):
            raise TypeError(
                "Wrong data type for parameter sgd_model. Only Sklearn SGDRegressor/Classifier data type is allowed.")

        if not isinstance(features, Iterable):
            raise TypeError("Wrong data type for parameter features. Only iterable data type is allowed.")

        for f in features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single features. Only string data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        # extract the parameters from the SGDRegressor/Classifer model
        params = SGDModelSQL.get_params(sgd_model)
        weights = params["weights"]
        bias = params["bias"]

        # create the SQL query that implements the SGDRegressor/Classifier inference
        pre_inference_query = None
        subquery = SGDModelSQL._sql_regression_part1(weights, features, table_name, dbms=self.dbms)
        query = SGDModelSQL._sql_regression_part2(bias, features, "({}) AS F ".format(subquery), dbms=self.dbms)

        return pre_inference_query, query
