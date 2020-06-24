from sklearn.preprocessing import OneHotEncoder
from collections import Iterable

class OneHotEncoderSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn OneHotEncoder object.
    """

    def __init__(self):
        self.params = None

    def _get_query_dense_ohe(self, table_name):
        """
        This method creates the SQL query that implements a dense one-hot-encoding transformation.

        :param table_name: the table name or the previous subquery where to read the data
        :return: the SQL query that implements a dense one-hot-encoding transformation
        """

        ohe_map = self.params["ohe_encoding"]
        remaining_features = self.params["other_features"]

        ohe_query = "SELECT "

        # implement one-hot encoding in SQL
        for ohe_feature_map in ohe_map:
            feature_after_ohe = ohe_feature_map["feature_after_ohe"]
            feature_before_ohe = ohe_feature_map["feature_before_ohe"]
            value = ohe_feature_map["value"]

            ohe_query += "CASE WHEN {} = '{}' THEN 1 ELSE 0 END AS {},\n".format(feature_before_ohe, value,
                                                                                 feature_after_ohe)

        # add the remaining features to the selection
        for f in remaining_features:
            ohe_query += "{},".format(f)
        ohe_query = ohe_query[:-1] # remove the last ','

        ohe_query += " FROM {}".format(table_name)

        return ohe_query

    def get_params(self, ohe, ohe_features, all_features, prev_transform_features=None):
        """
        This method extracts from the Sklearn One Hot Encoder all the fitted parameters needed to replicate in SQL the
        One Hot Encoding transformation.

        :param ohe: the fitted Sklearn's OneHotEncoder object
        :param ohe_features: the features to be one-hot-encoded
        :param all_features: all the feature names
        :param prev_transform_features: (optional) list of features transformed by previous transfomer
        :return: the ohe encoding extracted from the fitted Sklearn One Hot Encoder
        """

        if not isinstance(ohe, OneHotEncoder):
            raise TypeError("Wrong data type for parameter ohe. Only Sklearn's OneHotEncoder data type is allowed.")

        if not isinstance(ohe_features, Iterable):
            raise TypeError("Wrong data type for parameter ohe_features. Only iterable objects are allowed.")

        for f in ohe_features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single ohe_features. Only string data type is allowed.")

        if not isinstance(all_features, Iterable):
            raise TypeError("Wrong data type for parameter all_features. Only iterable objects are allowed.")

        for f in all_features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single all_features. Only string data type is allowed.")

        if prev_transform_features is not None:
            if not isinstance(prev_transform_features, Iterable):
                raise TypeError(
                    "Wrong data type for parameter prev_transform_features. Only iterable objects are allowed.")

            for f in prev_transform_features:
                if not isinstance(f, str):
                    raise TypeError(
                        "Wrong data type for single prev_transform_features. Only string data type is allowed.")

        # get the output ohe-encoded feature names
        features_after_ohe = ohe.get_feature_names()

        ohe_map = []
        # loop over the categorical features obtained after the application of the Sklearn's One Hot Encoder
        for feature_after_ohe in features_after_ohe:
            # the categorical features after the Sklearn OHE follow the format x<column_id>_<column_val> (e.g., x1_a)
            feature_item = feature_after_ohe.split("_")
            # get categorical feature name
            feature = ohe_features[int(feature_item[0].replace('x', ""))]
            # get categorical feature val
            value = feature_item[1]

            ohe_mapping = {"feature_after_ohe": feature_after_ohe, "feature_before_ohe": feature, "value": value}
            ohe_map.append(ohe_mapping)

        if prev_transform_features is None:
            prev_transform_features = []

        remaining_features = []
        for f in all_features:
            if f in prev_transform_features or f in ohe_features:
                continue
            remaining_features.append(f)

        out_features = prev_transform_features + list(features_after_ohe) + remaining_features

        self.params = {"ohe_encoding": ohe_map, "out_all_features": out_features, "ohe_features": ohe_features,
                       "other_features": prev_transform_features + remaining_features,
                       'out_transform_features': list(features_after_ohe)}

        return self.params

    def query(self, table_name):
        """
        This method creates the SQL query that implements into SQL an One Hot Encoding.

        :param table_name: the table name or the previous subquery where to read the data
        :return: the SQL query that implements the One Hot Encoding
        """

        if not self.params:
            raise Exception("No parameters extracted from the fitted OneHotEncoder. Invoke the get_params method.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        # create the SQL query that performs the One Hot Encoding
        ohe_query = self._get_query_dense_ohe(table_name)

        return ohe_query
