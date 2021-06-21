from lightning.regression import SDCARegressor
from collections import Iterable
from utils.dbms_utils import DBMSUtils


class SDCARegressorSQL(object):
    """
    This class implements the SQL wrapper for the Lightning's SDCARegressor.
    """

    def __init__(self):
        self.dbms = None
        self.mode = None
        self.available_modes = ['dense', 'sparse']

        self.temp_table_name = 'sdca_table'
        self.temp_table_pk = 'fidx'
        self.temp_table_weight_col = 'weight'

    @staticmethod
    def check_params(params: dict):
        assert isinstance(params, dict), "Wrong data type for param 'params'."
        param_names = ['weights', 'bias']
        assert all([p in params for p in param_names]), "Wrong data format for param 'params'."

        return params

    def set_mode(self, mode: str):
        assert isinstance(mode, str), "Wrong data type for param 'mode'."
        assert mode in self.available_modes

        self.mode = mode

    def set_dbms(self, dbms: str):
        self.dbms = dbms

    def reset_optimization(self):
        pass

    @staticmethod
    def get_params(sdca_model):
        """
        This method extracts from the SDCARegressor the fitted parameters (i.e., weights and intercept)
        :return: Python dictionary containing the fitted parameters extracted from the Lightning's SDCARegressor
        """

        weights = sdca_model.coef_.T.ravel()
        bias = 0
        if hasattr(sdca_model, "intercept_"):
            bias = sdca_model.intercept_
            # bias = sdca_model.intercept_[0]

        return {"weights": weights, "bias": bias}

    @staticmethod
    def _sql_regression_part1(weights, columns, table_name, dbms: str):
        """
        This method creates the portion of the SQL query responsible for the application of the dot product between
        regression weights and features.

        :param weights: the regression weights
        :param columns: the feature names
        :param table_name: the name of the table where read the data
        :param dbms: the name of the dbms
        :return: the portion of the SQL query which implements regression dot products
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

        query = query[:-1]  # remove the last ',ì
        query += " FROM {}".format(table_name)

        return query

    @staticmethod
    def _sql_regression_part2(bias, columns, table_name, dbms: str):
        """
        This method creates the portion of the SQL query responsible for the application of the linear combination over
        the regression dot products.

        :param bias: the regression bias
        :param columns: the feature names
        :param table_name: the table name or subquery where to read the data
        :param dbms: the name of the dbms
        :return: the portion of the SQL query which implements the regression dot product linear combination
        """

        if not isinstance(bias, (int, float)):
            raise TypeError("Wrong data type for parameter bias. Only numeric data type is allowed.")

        if not isinstance(columns, Iterable):
            raise TypeError("Wrong data type for parameter columns. Only iterable data type is allowed.")

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
        query += ") AS Score,"
        query = query[:-1]

        query += " FROM {}".format(table_name)

        return query

    def create_temp_table(self, params: dict):
        """
        This method creates the table that will store the regression parameters (i.e., weights and bias)

        :param params: the sdca regressor params
        :return: sql query containing the statement for the regression table creation
        """

        params = SDCARegressorSQL.check_params(params)

        create_stm = f'DROP TABLE IF EXISTS {self.temp_table_name};\n'
        create_stm += f'CREATE TABLE {self.temp_table_name} ({self.temp_table_pk} int, {self.temp_table_weight_col} float);\n'

        insert_stm = f'INSERT INTO {self.temp_table_name} VALUES (\n'
        for widx, w in enumerate(params['weights']):
            if self.dbms == 'sqlserver':
                if widx > 0 and widx % 1000 == 0:
                    insert_stm = insert_stm[:-2]  # remove ',\n'
                    insert_stm += ';\n\n'
                    insert_stm += f'INSERT INTO {self.temp_table_name} VALUES\n'
            insert_stm += f'({widx}, {w}),\n'
        insert_stm = insert_stm[-2:]    # remove ',\n'
        insert_stm += ');\n'

        index_name = f'{self.temp_table_name}_{self.temp_table_pk}'
        index_stm = DBMSUtils.create_index(self.dbms, index_name, self.temp_table_name, self.temp_table_pk)

        # query = f'{create_stm}{insert_stm}{index_stm}'

        return [create_stm, insert_stm, index_stm]

    def get_rolled_query(self, params, input_table: str, input_table_cols: dict, regression_table: str):
        """
        This method creates the SQL query that implements the SDCA regression function in a rolled version. This means
        that the query will read the data from an already created table where are stored SDCA weights and bias and will
        join these scores with the feature values deriving from the input table.

        :param params: the sdca regressor params
        :param input_table: the table or the previous subquery where the data will be read
        :param input_table_cols: the names of the columns of the input table
        :param regression_table: the table that contains the regression parameters
        :return: the SQL query that implements the regression
        """

        params = SDCARegressorSQL.check_params(params)
        assert isinstance(input_table, str), "Wrong data type for param 'input_table'."
        assert isinstance(input_table_cols, dict), "Wrong data type for param 'input_table_cols'."
        assert isinstance(regression_table, str), "Wrong data type for param 'regression_table'."
        assert 'pk' in input_table_cols, "Wrong data format for param 'input_table_cols'."
        assert 'fval' in input_table_cols, "Wrong data format for param 'input_table_cols'."
        assert 'fidx' in input_table_cols, "Wrong data format for param 'input_table_cols'."

        selection = " ( {} * {} ) AS prod ".format(input_table_cols['fval'], self.temp_table_weight_col)

        query = "SELECT {} FROM".format(selection)

        query += "({}) AS F \n INNER JOIN {} ON ({}={})".format(input_table, regression_table, input_table_cols['fidx'],
                                                                self.temp_table_pk)

        sub_query = "SELECT (SUM(prod) + {} ) AS Score\n FROM ({}) AS F GROUP BY {};".format(params['bias'], query,
                                                                                             input_table_cols['pk'])

        return sub_query

    def query(self, sdca_model: SDCARegressor, features, table_name: str):
        """
        This method creates the SQL query that performs the prediction with a SDCARegressor.

        :param sdca_model: the fitted Sklearn LogisticRegression object
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the SDCA regression inference
        """

        if not isinstance(sdca_model, SDCARegressor):
            raise TypeError(
                "Wrong data type for parameter sdca_regressor. Only Lightning SDCARegressor data type is allowed.")

        if table_name is not None:
            if not isinstance(table_name, str):
                raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        params = SDCARegressorSQL.get_params(sdca_model)
        weights = params["weights"]
        bias = params["bias"]

        pre_inference_queries = None
        if self.mode == 'dense':
            # create the SQL query that performs the prediction with a SDCARegressor
            subquery = SDCARegressorSQL._sql_regression_part1(weights, features, table_name, dbms=self.dbms)
            query = SDCARegressorSQL._sql_regression_part2(bias, features, "({}) AS F ".format(subquery),
                                                           dbms=self.dbms)

        elif self.mode == 'sparse':
            # FIXME check that the features are the column of triplet
            pre_inference_queries = self.create_temp_table(params)
            query = self.get_rolled_query(params, table_name, features, self.temp_table_name)

        else:
            raise ValueError("Wrong mode.")

        return pre_inference_queries, query
