from sklearn.preprocessing import OneHotEncoder
from collections import Iterable
from utils.dbms_utils import DBMSUtils
import logging
import random


class OneHotEncoderSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn OneHotEncoder object.
    """

    def __init__(self):
        self.params = None
        self.dbms = None
        self.available_mode = ['sparse', 'dense']
        self.mode = None
        self.input_table_name = None

        self.ohe_table_name = 'ohe_table'
        self.ohe_table_pk = 'id'
        self.ohe_table_fval_col = 'fval'
        self.ohe_table_fidx_col = 'fidx'

    def get_table_name(self):
        return self.ohe_table_name

    def get_table_cols(self):
        return {
            'pk': self.ohe_table_pk,
            'fidx': self.ohe_table_fidx_col,
            'fval': self.ohe_table_fval_col,
        }

    def set_mode(self, mode: str):
        assert isinstance(mode, str), "Wrong data type for param 'mode'."
        assert mode in self.available_mode, f"No mode {mode} found. Select one of {self.available_mode}."

        self.mode = mode

    def set_dbms(self, dbms: str):
        self.dbms = dbms

    @staticmethod
    def check_ohe_params(ohe_params: dict):
        assert isinstance(ohe_params, dict), "Wrong data type for param 'ohe_params'."
        param_names = ["ohe_encoding", "out_all_features", "ohe_features", "other_features", "out_transform_features",
                       'ohe2idx_map']
        err_msg = "Wrong format for param 'ohe_params'."
        assert all([p in ohe_params for p in param_names]), err_msg
        assert isinstance(ohe_params['ohe_encoding'], dict), err_msg
        assert isinstance(ohe_params['out_all_features'], list), err_msg
        assert isinstance(ohe_params['ohe_features'], list), err_msg
        assert isinstance(ohe_params['other_features'], list), err_msg
        assert isinstance(ohe_params['out_transform_features'], list), err_msg
        assert isinstance(ohe_params['ohe2idx_map'], dict), err_msg

        return ohe_params

    @staticmethod
    def _get_query_dense_ohe(ohe_params: dict, table_name: str, dbms: str):
        """
        This method creates the SQL query that implements a dense one-hot-encoding transformation.

        :param ohe_params: dictionary containing the parameters extracted from the fitted OneHotEncoder
        :param table_name: the table name or the previous subquery where to read the data
        :param dbms: the name of the dbms
        :return: the SQL query that implements a dense one-hot-encoding transformation
        """

        ohe_params = OneHotEncoderSQL.check_ohe_params(ohe_params)
        assert isinstance(table_name, str)
        assert isinstance(dbms, str)

        ohe_map = ohe_params["ohe_encoding"]
        remaining_features = ohe_params["other_features"]

        dbms_utils = DBMSUtils()

        ohe_query = "SELECT "

        # implement one-hot encoding in SQL
        for feature_after_ohe in ohe_map:
            # feature_after_ohe = ohe_feature_map["feature_after_ohe"]
            fao = dbms_utils.get_delimited_col(dbms, feature_after_ohe)
            ohe_feature_map = ohe_map[feature_after_ohe]
            feature_before_ohe = dbms_utils.get_delimited_col(dbms, ohe_feature_map["feature_before_ohe"])
            value = ohe_feature_map["value"]

            ohe_query += "CASE WHEN {} = '{}' THEN 1 ELSE 0 END AS {},\n".format(feature_before_ohe, value, fao)

        # add the remaining features to the selection
        for f in remaining_features:
            ohe_query += "{},".format(dbms_utils.get_delimited_col(dbms, f))
        ohe_query = ohe_query[:-1]  # remove the last ','

        ohe_query += " FROM {}".format(table_name)

        return ohe_query

    def _map_ohe_features_to_index(self, prev_features, ohe_features, next_features, original_ohe_features):
        """
        This method creates a dictionary that maps the input features (organized into 3 sets: previous non-categorical
        features, ohe features and next non-categorical features) with a progressive positional index. This index will
        indicate the new feature position after the application of the one hot encoding transformation.

        :param prev_features: non-categorical features that appear in the original dataset before the OHEed features
        :param ohe_features: OHEed features extracted from the fitted Sklearn's One Hot Encoder
        :param next_features: non-categorical features that appear in the original dataset after the OHEed features
        :param original_ohe_features: original names of the categorical features where the OHE has been applied
        :return: Python dictionary that associates to each feature the new index column inside the ohe-encoded dataset
        """

        assert isinstance(prev_features, Iterable), "Wrong data type for param 'prev_features'."
        if len(prev_features) > 0:
            assert all([isinstance(f, str) for f in prev_features])
        assert isinstance(ohe_features, Iterable), "Wrong data type for param 'ohe_features'."
        assert len(ohe_features) > 0, "Empty ohe features."
        assert all([isinstance(f, str) for f in ohe_features])
        assert isinstance(next_features, Iterable), "Wrong data type for param 'next_features'."
        if len(next_features) > 0:
            assert all([isinstance(f, str) for f in next_features])
        assert isinstance(original_ohe_features, Iterable), "Wrong data type for param 'original_ohe_features'."
        assert len(original_ohe_features) > 0, "Empty original ohe features."

        ohe_to_index_map = {}

        # loop over the previous non-categorical features and save them into the dictionary
        for prevf_idx, prevf in enumerate(prev_features):
            ohe_to_index_map[prevf] = {'': prevf_idx}
        initial_bias = len(prev_features)

        # loop over the categorical features obtained after the application of the Sklearn's One Hot Encoder and exploit
        # their positions to determine the new column index inside the transformed dataset
        for findex, ohe_feature in enumerate(ohe_features):

            # the categorical features after the Sklearn OHE follow the format x<column_id>_<column_val> (e.g., x1_a)
            feature_item = ohe_feature.split("_")
            # get categorical feature index
            feature_index_after_ohe = int(feature_item[0].replace('x', ""))
            # get categorical feature name
            ohe_feature_name = original_ohe_features[feature_index_after_ohe]
            # get categorical feature val
            ohe_feature_value = feature_item[1]

            # add the feature to the dictionary and assign to it a progressive index
            if ohe_feature_name not in ohe_to_index_map:
                ohe_to_index_map[ohe_feature_name] = {ohe_feature_value: findex + initial_bias}
            else:
                ohe_to_index_map[ohe_feature_name][ohe_feature_value] = findex + initial_bias

        ohe_feature_index = len(ohe_features)

        # loop over the next non-categorical features and save them into the dictionary
        for nextf_idx, nextf in enumerate(next_features):
            ohe_to_index_map[nextf] = {'': nextf_idx + ohe_feature_index}

        return ohe_to_index_map

    def get_params(self, ohe, ohe_features, all_features, prev_transform_features=None):
        """
        This method extracts from the Sklearn One Hot Encoder all the fitted parameters needed to replicate in SQL the
        One Hot Encoding transformation.

        :param ohe: the fitted Sklearn's OneHotEncoder object
        :param ohe_features: the features to be one-hot-encoded
        :param all_features: all the feature names
        :param prev_transform_features: (optional) list of features transformed by previous transformer
        :return: the ohe encoding extracted from the fitted Sklearn One Hot Encoder
        """

        assert isinstance(ohe, OneHotEncoder), "Wrong data type for param 'ohe'."
        assert isinstance(ohe_features, Iterable), "Wrong data type for param 'ohe_features'."
        assert all([isinstance(f, str) for f in ohe_features])
        assert isinstance(all_features, Iterable), "Wrong data type for param 'all_features'."
        assert all([isinstance(f, str) for f in all_features])
        if prev_transform_features is not None:
            assert isinstance(prev_transform_features, Iterable), "Wrong data type for param 'prev_transform_features'."
            assert all([isinstance(f, str) for f in prev_transform_features])

        # get the output ohe-encoded feature names
        features_after_ohe = ohe.get_feature_names()

        ohe_map = {}
        # loop over the categorical features obtained after the application of the Sklearn's One Hot Encoder
        for feature_after_ohe in features_after_ohe:
            # the categorical features after the Sklearn OHE follow the format x<column_id>_<column_val> (e.g., x1_a)
            feature_item = feature_after_ohe.split("_")
            # get categorical feature name
            feature = ohe_features[int(feature_item[0].replace('x', ""))]
            # get categorical feature val
            value = feature_item[1]

            ohe_map[feature_after_ohe] = {"feature_before_ohe": feature, "value": value}

        if prev_transform_features is None:
            prev_transform_features = []

        remaining_features = []
        for f in all_features:
            if f in prev_transform_features or f in ohe_features:
                continue
            remaining_features.append(f)

        out_features = prev_transform_features + list(features_after_ohe) + remaining_features

        ohe_to_index_map = self._map_ohe_features_to_index(prev_transform_features, features_after_ohe,
                                                           remaining_features, ohe_features)

        self.params = {"ohe_encoding": ohe_map, "out_all_features": out_features, "ohe_features": ohe_features,
                       "other_features": prev_transform_features + remaining_features,
                       'out_transform_features': list(features_after_ohe), 'ohe2idx_map': ohe_to_index_map}

        return self.params

    def _ohe_case_stm_encoded_values(self, ohe_map: dict, original_ohe_features: list):
        """
        This method creates the SQL case statement used in the sparse ohe query for emitting in output the values of the
        encoded features (i.e., the features encoded with a One Hot Encoding or with a previous normalization). If the
        case statement is not needed the encoded value is directly emitted.

        :param ohe_map: dictionary containing all or part of the ohe feature encodings
        :param original_ohe_features: all the categorical features of the dataset where the OHE has been applied
        :return: string containing the SQL case statement used in the sparse ohe query for emitting in output the values
                 of the encoded features
        """

        assert isinstance(ohe_map, dict), "Wrong data type for param 'ohe_map'."
        assert len(ohe_map) > 0, "Empty ohe map."
        assert isinstance(original_ohe_features, list), "Wrong data type for param 'original_ohe_features'."
        assert len(original_ohe_features) > 0, "Empty original ohe features."

        query = ""

        # get the categorical and numerical features included in the ohe_map
        cat_features = []
        num_features = []
        for feature in ohe_map:
            if feature in original_ohe_features:
                cat_features.append(feature)
            else:
                num_features.append(feature)

        assert len(cat_features) + len(num_features) == 1

        if len(cat_features) > 0:  # categorical feature
            query += '1'
        else:  # numerical feature
            query += num_features[0]

        return query

    def _ohe_case_stm_feature_indexes(self, ohe_map: dict, original_ohe_features: list):
        """
        This method creates the SQL case statement used in the sparse ohe query to assign to each encoded feature the
        new index position inside the ohe-encoded dataset.

        :param ohe_map: dictionary containing all or part of the ohe feature encodings
        :param original_ohe_features: all the categorical features of the dataset where the OHE has been applied
        :return: string containing the SQL case statement used in the sparse ohe query to assign to each encoded feature
                 the new index position inside the ohe-encoded dataset
        """

        assert isinstance(ohe_map, dict), "Wrong data type for param 'ohe_map'."
        assert len(ohe_map) > 0, "Empty ohe map."
        assert isinstance(original_ohe_features, list), "Wrong data type for param 'original_ohe_features'."
        assert len(original_ohe_features) > 0, "Empty original ohe features."

        # if the ohe is applied directly on the original table and the user-provided mapping includes only a single
        # numerical feature, no CASE statement is needed, but the new index column is directly emitted
        ohe_first_col = list(ohe_map)[0]
        if len(ohe_map) == 1 and ohe_first_col not in original_ohe_features:
            numerical_col_index = list(ohe_map[ohe_first_col].values())[0]
            return '{}'.format(numerical_col_index)

        # this query uses a COALESCE statement for managing data never seen during the training
        query = "COALESCE(CASE \n"

        # loop over the features and for each of them the new column's index is emitted
        ohe_columns = list(ohe_map.keys())
        for col in ohe_columns:
            col_map = ohe_map[col]

            if col in original_ohe_features:  # categorical feature
                # loop over categorical values and retrieve the new column index
                for cat_col_val in col_map:
                    cat_col_index = col_map[cat_col_val]

                    query += "WHEN {} = '{}' THEN {}\n".format(col, cat_col_val, cat_col_index)

            else:  # numerical feature
                # numerical features have not been ohe-encoded during the Sklearn training so there is only a value
                # associated to these features
                assert len(col_map) == 1

        # manage never seen data during the training

        # a COALESCE statement is used for managing data never seen during the training
        # the value used in the COALESCE statement to replace the null value has to be unique in a such a way that it
        # won't match with any feature index of the next component in the ml pipeline
        # for this reason it is obtained by summing the bias 99999999 with an number sampled in the range (0, 99999999)
        rand_idx = 99999999 + random.randint(0, 99999999)
        assert len(ohe_columns) == 1
        subquery_never_seen_data = f'{rand_idx}'

        query += " END, {})\n".format(subquery_never_seen_data)

        return query

    def _create_ohe_query(self, ohe_map: dict, original_ohe_features: list, batch_mode: bool = False):
        """
        This method creates the SQL query that implements a sparse One Hot Encoding transformation over the provided
        features (i.e., the ohe_map param). If the provided features don't cover all the ohe features, a batch mode
        modality has to be activated. In this case the query performs a partial One Hot Encoding by considering a
        vertical partition of the data.

        The OHE SQL query uses two case statements:
        1) the first is used to emit the encoded values: the encoded value for the OHEed columns is 1, while the one for
        not OHEed columns (e.g., the numerical features) corresponds to their original value
        2) the second case statement provides the index of non-zero values
        Note that indexes are sequential, even across categorical columns. This is because we implicitly concatenate
        one-hot encoded columns into a unique feature vector.

        :param ohe_map: dictionary containing all or part of the ohe feature encodings
        :param original_ohe_features: all the categorical features of the dataset where the OHE has been applied
        :param batch_mode: boolean flag that indicates whether the provided ohe map covers a subset of the ohe encodings
        :return: the SQL query that implements a sparse One Hot Encoding transformation over the provided ohe features
        """

        # check input parameter data types
        assert isinstance(ohe_map, dict), "Wrong data type for param 'ohe_map'."
        assert len(ohe_map) > 0, "Empty ohe map."
        assert isinstance(original_ohe_features, list), "Wrong data type for param 'original_ohe_features'."
        assert len(original_ohe_features) > 0, "Empty original ohe features."
        assert isinstance(batch_mode, bool), "Wrong data type for param 'batch_mode'."

        # get the categorical and numerical features included in the ohe_map
        cat_features = []
        num_features = []
        for feature in ohe_map:
            if feature in original_ohe_features:
                cat_features.append(feature)
            else:
                num_features.append(feature)

        # SQL case statement 1
        case_values = self._ohe_case_stm_encoded_values(ohe_map, original_ohe_features)

        # SQL case statement 2
        case_indexes = self._ohe_case_stm_feature_indexes(ohe_map, original_ohe_features)

        # add the where clause in the output query
        where_clause = "WHERE "

        # if the batch mode is active then only the provided features are extracted from the table
        if batch_mode:

            # select from the input table only the tuples that contain (for the considered categorical features) the
            # values included in the ohe_map
            for cat_feature in cat_features:
                cat_feature_vals = list(ohe_map[cat_feature].keys())
                cat_feature_vals_list = ', '.join(["'{}'".format(v) for v in cat_feature_vals])

                where_clause += "{} IN ({}) OR\n".format(cat_feature, cat_feature_vals_list)

            if len(cat_features) > 0:
                if len(num_features) == 0:
                    where_clause = where_clause[:-4]

            if len(cat_features) == 0:
                if where_clause == "WHERE ":  # user hasn't provided any where clause
                    where_clause = ""
                else:  # user has provided any where clause
                    where_clause = where_clause[:-5]  # remove ' AND '

        else:
            if where_clause == "WHERE ":
                where_clause = ""
            else:
                where_clause = where_clause[:-5]  # remove ' AND '

        ohe_query = f"SELECT {self.ohe_table_pk}, {case_values} AS {self.ohe_table_fval_col}, " \
                    f"{case_indexes} AS {self.ohe_table_fidx_col}\n"

        from_clause = ""
        # if the ohe is applied directly on the original table, one column at a time is processed; let's project
        # the selection on the considered column
        assert len(cat_features) + len(num_features) == 1

        if len(cat_features) == 1:
            target_col = cat_features[0]
        else:
            target_col = num_features[0]

        from_clause += f"( SELECT {self.ohe_table_pk}, {target_col} FROM {self.input_table_name} {where_clause} )"

        ohe_query += " FROM {} AS F;\n\n".format(from_clause)

        return ohe_query

    def _get_query_sparse_ohe(self, ohe_params: dict):
        """
        This method creates an SQL query that implements a sparse ohe transformation.
        The query is composed by a main CASE statement that replicates the OHE mapping.
        For high dimensional data, it is not possible to encode the mapping inside a single CASE statement, because of
        the limit of the number of WHEN statements that can be inserted in a query. In this case multiple queries are
        generated and for each of them the maximum number of WHEN statements allowed is considered. Each query result is
        saved into a temporary table.

        :param ohe_params: dictionary containing the parameters extracted from the fitted OneHotEncoder
        :return: the SQL query that implements the sparse One Hot Encoding transformation
        """

        ohe_params = OneHotEncoderSQL.check_ohe_params(ohe_params)
        ohe_feature_map = ohe_params['ohe2idx_map']
        original_ohe_features = ohe_params['ohe_features']

        ohe_query = ""

        # considering that each feature after the ohe is used to create a WHEN statement, it is needed to check if this
        # number if greater (or not) than the maximum number of WHEN statements that can be included in a single CASE
        # statement. For SQLSERVER the maximum number of WHEN statements is 9700.
        # https://www.sqlservercentral.com/forums/topic/maximum-number-of-when-then-lines-in-a-case-statement
        sql_max_when_statements = 9700
        # sql_max_when_statements = 100

        # if the OHE is applied directly on the original table then a temporary table is created to store OHE
        # results in a triplet data format
        warn_message = "A temporary table 'ohe_table' will be created."
        logging.warning(warn_message)

        # add to the ohe query the SQL statement for the creation of the intermediate ohe table
        create_ohe_table_query = f"DROP TABLE IF EXISTS {self.ohe_table_name};\n"
        create_ohe_table_query += f"CREATE TABLE {self.ohe_table_name}({self.ohe_table_pk} int, "
        create_ohe_table_query += f"{self.ohe_table_fval_col} float, {self.ohe_table_fidx_col} int);\n\n"
        # create_ohe_table_query += f" PRIMARY KEY({self.ohe_table_pk}, {self.ohe_table_fidx_col}));\n\n"
        # ohe_query += create_ohe_table_query

        # split, if needed, the OHEed features in batches smaller than the SQL limits

        ohe_feature_map_batches = []
        num_batch = 1

        # loop over OHEed columns
        for col in ohe_feature_map:
            feature_map = ohe_feature_map[col]
            num_ohe_features_per_col = len(feature_map)

            # check if the number of features derived from the current OHEed column is greater than the DBMS limits
            if num_ohe_features_per_col > sql_max_when_statements:

                # split the query in multiple batch queries
                batch_size = sql_max_when_statements
                if num_ohe_features_per_col % batch_size == 0:
                    num_batch = num_ohe_features_per_col // batch_size
                else:
                    num_batch = num_ohe_features_per_col // batch_size + 1

                feature_map_vals = list(feature_map.items())
                # loop over the number of batch
                for i in range(num_batch):
                    # select a partition of the features after ohe
                    batch_ohe_feature_map = dict(feature_map_vals[i * batch_size:i * batch_size + batch_size])
                    ohe_feature_map_batches.append({col: batch_ohe_feature_map})

            else:
                ohe_feature_map_batches.append({col: feature_map})

        # loop over the batches
        ohe_sub_queries = []
        for ohe_feature_map_batch in ohe_feature_map_batches:

            # create the SQL query that applies the One Hot Encoding on the selected features
            batch_mode = False
            if num_batch > 1:
                batch_mode = True
            ohe_batch_query = self._create_ohe_query(ohe_feature_map_batch, original_ohe_features,
                                                     batch_mode=batch_mode)
            ohe_sub_queries.append(ohe_batch_query)

        # optimization: combine multiple ohe batch queries to reduce the total number of INSERT statements
        cum_sum = 0
        current_combined_suq_queries = []
        for j in range(len(ohe_feature_map_batches)):
            suq_query = ohe_sub_queries[j]
            ohe_feature_map_batch = ohe_feature_map_batches[j]
            ohe_batch_query_size = len(list(ohe_feature_map_batch.values())[0])

            cum_sum += ohe_batch_query_size
            current_combined_suq_queries.append(suq_query)

            if cum_sum > sql_max_when_statements:
                cum_sum = ohe_batch_query_size
                list_joint_sub_queries = current_combined_suq_queries[:-1]
                current_combined_suq_queries = [current_combined_suq_queries[-1]]

                joint_sub_queries = ""
                for sub_query in list_joint_sub_queries:
                    joint_sub_queries += "{}\n\n UNION ALL \n\n".format(sub_query[:-3])  # remove ';\n\n'
                joint_sub_queries = joint_sub_queries[:-15] + ";"  # remove '\n\n UNION ALL \n\n'

                # if multiple batch queries are generated, they have to be saved in a temporary table with an
                # INSERT statement
                insert_stm = ""
                if num_batch > 1:
                    insert_stm += f"INSERT INTO {self.ohe_table_name}\n"
                else:
                    insert_stm += f"INSERT INTO {self.ohe_table_name}\n"

                ohe_query += "{}{}\n\n".format(insert_stm, joint_sub_queries)

        # combine the last ohe sub queries
        joint_sub_queries = ""
        for sub_query in current_combined_suq_queries:
            joint_sub_queries += "{}\n\n UNION ALL \n\n".format(sub_query[:-3])  # remove ';\n\n'
        joint_sub_queries = joint_sub_queries[:-15] + ";"  # remove '\n\n UNION ALL \n\n'

        # if multiple batch queries are generated, they have to be saved in a temporary table with an
        # INSERT statement
        insert_stm = ""
        if num_batch > 1:
            insert_stm += f"INSERT INTO {self.ohe_table_name}\n"
        else:
            insert_stm += f"INSERT INTO {self.ohe_table_name}\n"

        ohe_query += "{}{}\n\n".format(insert_stm, joint_sub_queries)

        # create an index on the ohe table
        index_ohe = DBMSUtils.create_index(dbms=self.dbms, index_name=f'{self.ohe_table_name}_{self.ohe_table_pk}',
                                           target_table=self.ohe_table_name, target_col=self.ohe_table_pk)
        index_ohe += DBMSUtils.create_index(dbms=self.dbms,
                                            index_name=f'{self.ohe_table_name}_{self.ohe_table_fidx_col}',
                                            target_table=self.ohe_table_name, target_col=self.ohe_table_fidx_col)
        # ohe_query += index_ohe

        return [create_ohe_table_query, ohe_query, index_ohe], f'select * from {self.ohe_table_name}'

    def query(self, table_name):
        """
        This method creates the SQL query that implements into SQL an One Hot Encoding.

        :param table_name: the table name or the previous subquery where to read the data
        :return: the SQL query that implements the One Hot Encoding
        """

        assert isinstance(table_name, str), "Wrong data type for param 'table_name'."
        assert self.params is not None, "No ohe params extracted."
        assert self.mode is not None, "No mode selected."

        dbms_util = DBMSUtils()
        auto_inc = dbms_util.get_auto_increment_col(self.dbms)
        # if the table provided in input is the result of a previous query
        if len(table_name) > 7 and table_name[-7:] == 'AS data':
            real_tab_name = 'data'
        else:   # only a table name is provided
            real_tab_name = table_name
        self.input_table_name = f'(select {auto_inc} AS {self.ohe_table_pk}, {real_tab_name}.* FROM {table_name}) AS T'

        # create the SQL query that performs the One Hot Encoding
        pre_ohe_queries = None
        if self.mode == 'dense':
            ohe_query = self._get_query_dense_ohe(self.params, table_name, dbms=self.dbms)

        elif self.mode == 'sparse':
            pre_ohe_queries, ohe_query = self._get_query_sparse_ohe(self.params)

        else:
            raise ValueError(f"Wrong mode ({self.mode}).")

        return pre_ohe_queries, ohe_query
