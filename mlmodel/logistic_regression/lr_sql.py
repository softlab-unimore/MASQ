from collections import Iterable
from sklearn.linear_model import LogisticRegression
from utils.dbms_utils import DBMSUtils


class LogisticRegressionSQL(object):
    """
    This class implements the SQL wrapper for the Sklearn's LogisticRegression.
    """

    def __init__(self):
        self.dbms = None
        self.available_modes = ['dense', 'sparse']
        self.mode = None

        self.temp_table_name = 'logistic_table'
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
    def get_params(lr):
        """
        This method extracts from the LogisticRegression the fitted parameters (i.e., weights and intercepts for each
        class)

        :param lr: the fitted Sklearn LogisticRegression object
        :return: Python dictionary containing the fitted parameters extracted from the Sklearn's LogisticRegression
        """

        if not isinstance(lr, LogisticRegression):
            raise TypeError("Wrong data type for parameter lr. Only Sklearn LogisticRegression data type is allowed.")

        weights = lr.coef_
        bias = lr.intercept_

        return {"weights": weights, "bias": bias}

    def _get_linear_combination(self, weights, bias, columns):
        """
        This method generates the linear combination component of the LogisticRegression function.

        :param weights: the weights for a target class
        :param bias: the bias for a target class
        :param columns: the feature names
        :return: the portion of SQL query responsible for the application of the linear combination component of the
                 LogisticRegression function
        """

        dbms_util = DBMSUtils()

        query = ""
        for i in range(len(columns)):
            c = dbms_util.get_delimited_col(self.dbms, columns[i])
            query += "({}*{}) + ".format(c, weights[i])
        query = "{} {}".format(query, bias)

        return query

    def _get_rolled_linear_combination(self, bias, input_table: str, input_table_cols: dict):
        """
        This method generates the linear combination component of the LogisticRegression function in a rolled version.
        This means that the linear combination is computed by reading the data from temporary tables.

        :param bias: the LogisticRegression bias
        :param input_table: the table or the previous subquery where the data will be read
        :param input_table_cols: the names of the columns of the input table
        :return: the SQL query that implements the regression
        """

        selection = " ( {} * {} ) AS prod ".format(input_table_cols['fval'], self.temp_table_weight_col)

        query = "SELECT {} FROM".format(selection)

        query += "({}) AS F \n INNER JOIN {} ON ({}={})".format(input_table, self.temp_table_name,
                                                                input_table_cols['fidx'], self.temp_table_pk)

        sub_query = "SELECT (SUM(prod) + {} ) AS Score\n FROM ({}) AS F GROUP BY {}".format(bias, query,
                                                                                            input_table_cols['pk'])

        return sub_query

    def _get_raw_query(self, params, features, table_name, class_labels):
        """
        This method creates the SQL query responsible for the application of the Logistic Regression function.

        :param params: LogisticRegression params
        :param features: the feature names
        :param table_name: the name of the table or the subquery where to read the data
        :param class_labels: the labels of the class attribute
        :return: the SQL query responsible for the application of the Logistic Regression function
        """

        params = LogisticRegressionSQL.check_params(params)
        weights = params['weights']
        bias = params['bias']
        assert isinstance(features, (Iterable, dict)), "Wrong data type for param 'features'."
        if table_name is not None:
            assert isinstance(table_name, str), "Wrong data type for param 'table_name'."
        assert isinstance(class_labels, Iterable), "Wrong data type for param 'class_labels'."
        if self.mode == 'sparse':
            assert isinstance(features, dict), "Wrong data type for param 'features'."
            f_params = ['pk', 'fidx', 'fval']
            assert all([p in features for p in f_params]), "Wrong data format for param 'features'."
        else:
            assert isinstance(features, Iterable), "Wrong data type for param 'features'."
            assert all([isinstance(p, str) for p in features]), "Wrong data format for param 'features'."

        wildcard = "class_"
        query_internal = "SELECT "

        if self.mode == 'dense':

            for i in range(len(weights)):
                w = weights[i]
                b = bias[i]
                q = self._get_linear_combination(w, b, features)
                query_internal += "({}) AS {}{},".format(q, wildcard, i)

            query_internal = query_internal[:-1]  # remove the last ','
            query_internal += "\n FROM {}".format(table_name)
            query_internal = " ( {} ) AS F ".format(query_internal)

        else:

            # FIXME: the name data for the input table derive from outside this class
            query_internal += ' F.id, '
            for i in range(len(weights)):
                b = bias[i]
                query_internal += f'(SUM(prod_{i}) + {b}) as {wildcard}{i}, '
            query_internal = query_internal[:-2]    # remove ', '
            query_internal += ' FROM ('
            query_internal += 'SELECT data.id, '
            for i in range(len(weights)):
                tmp_table_name = f'{self.temp_table_name}_{i}'
                query_internal += f"(data.{features['fval']} * {tmp_table_name}.{self.temp_table_weight_col}) AS prod_{i}, "
            query_internal = query_internal[:-2]  # remove ', '
            query_internal += f' FROM {table_name}\n'
            query_internal += f"INNER JOIN {tmp_table_name} ON (data.{features['fidx']}={tmp_table_name}.{self.temp_table_pk})\n"
            query_internal += ') AS F\n'
            query_internal += f"GROUP BY F.{features['pk']}"

            query_internal = " ( {} ) AS F ".format(query_internal)

        query = "SELECT "

        if len(class_labels) > 2:

            sum_query = "("
            for i in range(len(bias)):
                sum_query += "EXP({}{})+".format(wildcard, i)
            sum_query = sum_query[:-1]  # remove the last '+'
            sum_query += ")"

            for i in range(len(bias)):
                query += "(" + "EXP({}{}) / {} ) AS {}{},".format(wildcard, i, sum_query, wildcard, i)

            query = query[:-1]  # remove the last ','
            query = "{}\n FROM {}".format(query, query_internal)

            case_stm = "CASE"
            for i in range(len(class_labels)):
                case_stm += " WHEN "
                for j in range(len(class_labels)):
                    if j == i:
                        continue
                    case_stm += "{}{} >= {}{} AND ".format(wildcard, i, wildcard, j)
                case_stm = case_stm[:-5] # remove the last ' AND '
                case_stm += " THEN {}\n".format(class_labels[i])
            case_stm += "END AS Score"

            final_query = "SELECT {} FROM ({}) AS F".format(case_stm, query)
        else:
            if self.mode == 'sparse':
                score_to_class_query = "SELECT F.id, CASE WHEN -1.0*{}0 > 500 THEN 0 ELSE".format(wildcard)
            else:
                score_to_class_query = "SELECT CASE WHEN -1.0*{}0 > 500 THEN 0 ELSE".format(wildcard)
            score_to_class_query += " 1.0/(1.0+EXP(-1.0*{}0)) END AS PROB_0".format(wildcard)
            score_to_class_query += " FROM {}".format(query_internal)

            final_query = "SELECT CASE WHEN PROB_0 > (1-PROB_0) THEN 1 ELSE 0 END AS Score FROM ({}) AS F".format(
                score_to_class_query)

            if self.mode == 'sparse':
                final_query += ' ORDER BY F.id'

        return final_query

    def create_temp_table(self, params: dict):
        """
        This method creates the table that will store the logistic regression parameters (i.e., weights and bias)

        :param params: the logistic regression params
        :return: sql query containing the statement for the logistic regression table creation
        """

        params = LogisticRegressionSQL.check_params(params)

        query_list = []
        # query = ''
        for class_idx in range(len(params['weights'])):
            tab_name = f'{self.temp_table_name}_{class_idx}'
            class_weights = params['weights'][class_idx]

            create_stm = f'DROP TABLE IF EXISTS {tab_name};\n'
            create_stm += f'CREATE TABLE {tab_name} ({self.temp_table_pk} int, {self.temp_table_weight_col} float);\n'

            insert_stm = f'INSERT INTO {tab_name} VALUES\n'
            for widx, w in enumerate(class_weights):
                if self.dbms == 'sqlserver':
                    if widx > 0 and widx % 1000 == 0:
                        insert_stm = insert_stm[:-2]  # remove ',\n'
                        insert_stm += ';\n\n'
                        insert_stm += f'INSERT INTO {tab_name} VALUES\n'

                insert_stm += f'({widx}, {w}),\n'
            insert_stm = insert_stm[:-2]    # remove ',\n'
            insert_stm += ';\n'

            index_name = f'{tab_name}_{self.temp_table_pk}'
            index_stm = DBMSUtils.create_index(self.dbms, index_name, tab_name, self.temp_table_pk)

            # query += f'{create_stm}{insert_stm}{index_stm}'
            query_list += [create_stm, insert_stm, index_stm]

        return query_list

    def query(self, lr, features, table_name, class_labels=None):
        """
        This method creates the SQL query that performs the LogisticRegression inference.

        :param lr: the fitted Sklearn LogisticRegression object
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :param class_labels: the labels of the class attribute
        :return: the SQL query that implements the Logistic Regression inference
        """

        if not isinstance(lr, LogisticRegression):
            raise TypeError("Wrong data type for parameter lr. Only Sklearn LogisticRegression data type is allowed.")

        if not isinstance(features, Iterable):
            raise TypeError("Wrong data type for parameter features. Only iterable data type is allowed.")

        for f in features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single features. Only string data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        if class_labels is not None:
            if not isinstance(class_labels, Iterable):
                raise TypeError("Wrong data type for parameter class_labels. Only iterable data type is allowed.")

        params = LogisticRegressionSQL.get_params(lr)
        if not class_labels:
            class_labels = list(lr.classes_)

        assert self.mode is not None, "No mode selected."

        # generate the query for the temporary table creation if the rolled mode has been selected
        pre_inference_queries = None
        if self.mode == 'sparse':
            # FIXME check that the features are the column of triplet
            pre_inference_queries = self.create_temp_table(params)

        # create the SQL query that implements the LogisticRegression inference
        query = self._get_raw_query(params, features, table_name, class_labels)

        return pre_inference_queries, query
