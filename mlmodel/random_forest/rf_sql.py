from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mlmodel.decision_tree.dtc_sql import DTMSQL


class RFMSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn's Random Forest Model (RFM).

    The current random forest classifier implementation applies a majority voting technique across the forest labels,
    while Sklearn computes the the predicted class by selecting the one with highest mean probability estimate across
    the trees.
    """

    def __init__(self, classification: bool = False):
        """
        This method initializes the state of the Random Forest Model SQL wrapper.

        :param classification: boolean flag that indicates whether the RFM is used in classification or regression
        """

        assert isinstance(classification, bool), "Only boolean data type is allowed for param 'classification'."

        self.classification = classification
        self.nested = True
        self.merge_ohe_features = None

    def set_nested_implementation(self):
        self.nested = True

    def set_flat_implementation(self):
        self.nested = False

    @staticmethod
    def _check_merge_ohe_features(merge_ohe_features: dict):
        error_msg = "Wrong data format for parameter 'merge_ohe_features'."
        assert isinstance(merge_ohe_features, dict), error_msg
        params = ["feature_before_ohe", "value"]
        for key, item in merge_ohe_features.items():
            assert isinstance(key, str)
            assert isinstance(item, dict), error_msg
            assert all(p in params for p in item), error_msg
            assert isinstance(item["feature_before_ohe"], str), error_msg
            assert isinstance(item["value"], str), error_msg

    def merge_ohe_with_trees(self, merge_ohe_features: dict):
        DTMSQL._check_merge_ohe_features(merge_ohe_features)
        self.merge_ohe_features = merge_ohe_features

    def reset_optimization(self):
        self.merge_ohe_features = None

    @staticmethod
    def get_params(rfm: (RandomForestClassifier, RandomForestRegressor), features: list, is_classification: bool,
                   nested: bool, merge_ohe_features: dict = None):
        """
        This method extracts the tree rules from the Sklearn's Random Forest Model and creates their SQL representation.

        :param rfm: the fitted Sklearn Random Forest Model
        :param features: the list of features
        :param is_classification: boolean flag that indicates whether the RFM is used in classification or regression
        :param nested: boolean flag that indicates whether to use the nested SQL conversion technique
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: Python dictionary containing the parameters extracted from the fitted RFM
        """

        error_msg = "Only RandomForestClassifier/RandomForestRegressor type is allowed fo param 'rfm'."
        assert isinstance(rfm, (RandomForestClassifier, RandomForestRegressor)), error_msg

        # extract the rules from the RFM decision trees
        trees = rfm.estimators_
        trees_params = []

        # loop over the trees
        for index, tree in enumerate(trees):

            # extract the rules from the current tree
            tree_params = DTMSQL.get_params(tree, list(features), is_classification, nested,
                                            merge_ohe_features=merge_ohe_features)
            tree_params["weight"] = 1.0/len(trees)
            trees_params.append(tree_params)

        classes = None
        if is_classification:
            try:
                classes = rfm.classes_
            except AttributeError:
                raise AttributeError(
                    "No attribute classes_ found in the Random Forest Model. Is it a classifier?")

        rfm_params = {'classes': classes, 'trees_params': trees_params}

        return rfm_params

    def _rfm_to_sql(self, rfm_params: dict, table_name: str):
        """
        This method generates the SQL query that implements the RFM inference.

        :param rfm_params: the parameters extracted from the Sklearn's RFM object
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the RFM inference
        """

        assert isinstance(rfm_params, dict), "Only Python dictionary data type is allowed for param 'rfm_params'."
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."
        params_keys = ["classes", "trees_params"]
        assert all(p in params_keys for p in rfm_params), "Wrong data format for parameter 'rfm_params'."
        if self.classification:
            assert rfm_params['classes'] is not None
        tree_params_keys = ["estimator", "sql_rules", "weight"]
        for tree_params in rfm_params["trees_params"]:
            assert all(p in tree_params_keys for p in tree_params), "Wrong data format for parameter 'rfm_params'."

        classes = rfm_params["classes"]
        trees_params = rfm_params["trees_params"]

        query = "SELECT "
        # loop over the trees and create a CASE statement for each tree
        for i in range(len(trees_params)):
            tree_params = trees_params[i]
            sql_case = tree_params["sql_rules"]

            sql_case += " AS tree_{},".format(i)

            query += sql_case

        query = query[:-1] # remove the last ","

        query += " FROM {}".format(table_name)

        if self.classification: # classification task

            # find the majority class

            # count the number of trees that have predicted the same class label
            majority_class_query = "SELECT "
            for class_ix, class_label in enumerate(classes):
                majority_class_query += "("

                for i in range(len(trees_params)):
                    majority_class_query += "CASE WHEN tree_{} = {} THEN 1 ELSE 0 END + ".format(i, class_label)
                majority_class_query = majority_class_query[:-3] # remove the last ' + '

                majority_class_query += ") AS class_{}, ".format(class_ix)
            majority_class_query = majority_class_query[:-2] # remove the last ', '
            majority_class_query += " FROM ({}) AS F".format(query)

            # find the majority class label
            final_query = "SELECT "
            case_stm = "CASE"
            for i in range(len(classes)):
                case_stm += " WHEN "
                for j in range(len(classes)):
                    if j == i:
                        continue
                    case_stm += "class_{} >= class_{} AND ".format(i, j)
                case_stm = case_stm[:-5]  # remove the last ' AND '
                case_stm += " THEN {}\n".format(classes[i])
            case_stm += "END AS Score"

            final_query += "{} FROM ({}) AS F".format(case_stm, majority_class_query)

        else: # regression task

            # combine the tree scores with a mean
            external_query = " SELECT ("
            for i in range(len(trees_params)):
                tree_weight = trees_params[i]["weight"]
                external_query += "{} * tree_{} + ".format(tree_weight, i)

            external_query = external_query[:-3] # remove the last ' + '
            external_query += ") AS Score"

            # combine internal and external queries in a single query
            final_query = "{} FROM ({}) AS F".format(external_query, query)

        return final_query

    def query(self, rfm: (RandomForestClassifier, RandomForestRegressor), features: list, table_name: str):
        """
        This method generates the SQL query that implements the RFM inference.

        :param rfm: the fitted Sklearn's Random Forest Model
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the RFM inference
        """

        error_msg = "Only RandomForestClassifier/RandomForestRegressor type is allowed fo param 'rfm'."
        assert isinstance(rfm, (RandomForestClassifier, RandomForestRegressor)), error_msg
        assert isinstance(features, list), "Only list data type is allowed for param 'features'"
        for f in features:
            assert isinstance(f, str)
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."

        # extract the parameters (i.e., the decision rules) from the fitted RFM
        rfm_params = RFMSQL.get_params(rfm, features, is_classification=self.classification, nested=self.nested,
                                       merge_ohe_features=self.merge_ohe_features)

        # create the SQL query that implements the RFM inference
        query = self._rfm_to_sql(rfm_params, table_name)

        return query
