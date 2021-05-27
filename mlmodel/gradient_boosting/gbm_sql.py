from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from mlmodel.decision_tree.dtc_sql import DTMSQL


class GBMSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn's Gradient Boosting Model (GBM).
    """

    def __init__(self, classification: bool = False):
        """
        This method initializes the state of the Gradient Boosting Model SQL wrapper.

        :param classification: boolean flag that indicates whether the GBM is used in classification or regression
        """

        assert isinstance(classification, bool), "Only boolean data type is allowed for param 'classification'."

        self.init_score = 0
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
    def get_params(gbm: (GradientBoostingClassifier, GradientBoostingRegressor), features: list,
                   is_classification: bool, nested: bool, merge_ohe_features: dict = None):
        """
        This method extracts the tree rules from the Sklearn's Gradient Boosting Model and creates their SQL
        representation.

        :param gbm: the fitted Sklearn's Gradient Boosting model
        :param features: the list of features
        :param is_classification: boolean flag that indicates whether the GBM is used in classification or regression
        :param nested: boolean flag that indicates whether to use the nested SQL conversion technique
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: Python dictionary containing the parameters extracted from the fitted GBM
        """

        error_msg = "Only GradientBoostingClassifier/GradientBoostingRegressor type is allowed fo param 'gbm'."
        assert isinstance(gbm, (GradientBoostingClassifier, GradientBoostingRegressor)), error_msg

        # extract decision rules from the GBM decision trees
        trees = gbm.estimators_

        # loop over trees
        trees_params = []
        for index, tree in enumerate(trees):

            tree_params = []
            for idx, class_tree in enumerate(tree):

                # extract the rules from the current tree

                # a GradientBoosting classifier fits n_classes parallel decision tree regressor, so both in
                # classification and regression tasks the parameters of a generic regressor tree has to be extracted
                class_tree_params = DTMSQL.get_params(class_tree, list(features), is_classification=False,
                                                      nested=nested, merge_ohe_features=merge_ohe_features)
                class_tree_params["weight"] = gbm.learning_rate
                tree_params.append(class_tree_params)

            trees_params.append(tree_params)

        try:
            init_score = gbm.init_score
        except AttributeError:
            raise AttributeError("No init_score attribute provided in the fitted GBM object.")

        classes = None
        if is_classification:
            try:
                classes = gbm.classes_
            except AttributeError:
                raise AttributeError(
                    "No attribute classes_ found in the Gradient Boosting Model. Is it a classifier?")

        gbm_params = {"classes": classes, "init_score": init_score, "trees_params": trees_params}

        return gbm_params

    def _gbm_to_sql(self, gbm_params: dict, table_name: str):
        """
        This method generates the SQL query that implements the GBM inference.

        :param gbm_params: the parameters extracted from the Sklearn's GBM object
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the GBM inference
        """

        assert isinstance(gbm_params, dict), "Only Python dictionary data type is allowed for param 'gbm_params'."
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."
        params_keys = ["init_score", "classes", "trees_params"]
        assert all(p in params_keys for p in gbm_params), "Wrong data format for parameter 'gbm_params'."
        if self.classification:
            assert gbm_params['classes'] is not None
        assert isinstance(gbm_params["trees_params"], list)
        tree_params_keys = ["estimator", "sql_rules", "weight"]
        for class_tree_params in gbm_params["trees_params"]:
            assert isinstance(class_tree_params, list)
            assert len(class_tree_params) > 0
            if self.classification:
                if len(gbm_params['classes']) == 2:
                    assert len(class_tree_params) == 1
                else:
                    assert len(class_tree_params) == len(gbm_params['classes'])
            for tree_params in class_tree_params:
                assert all(p in tree_params_keys for p in tree_params), "Wrong data format for parameter 'gbm_params'."

        init_score = gbm_params['init_score']
        classes = gbm_params['classes']
        trees_params = gbm_params['trees_params']

        query = "SELECT "
        # loop over the trees and create a CASE statement for each tree
        for i in range(len(trees_params)):

            # in classification one tree for each class is fitted; loop over them
            class_tree_params = trees_params[i]
            for j in range(len(class_tree_params)):
                tree_params = class_tree_params[j]
                tree_weight = tree_params["weight"]
                sql_case = tree_params["sql_rules"]

                sql_case += " AS tree_{}_{},".format(i, j)

                query += sql_case

        query = query[:-1] # remove the last ","

        query += " FROM {}".format(table_name)

        # combine the tree scores for each class with a weighted sum
        n_classes = len(trees_params[0])
        external_query = " SELECT"
        for j in range(n_classes):
            external_query += " ("
            for i in range(len(trees_params)):
                class_tree_params = trees_params[i]
                tree_weight = class_tree_params[j]["weight"]
                external_query += "{} * tree_{}_{} + ".format(tree_weight, i, j)
                # external_query += "tree_{}_{} + ".format(i, j)

            external_query += "{} + ".format(init_score)

            external_query = external_query[:-2]
            external_query += ") AS Score_{},".format(j)
        external_query = external_query[:-1] # remove the last ','

        # combine internal and external queries in a single query
        comb_query = external_query + " FROM (" + query + " ) AS F"

        if self.classification: # the model is used in a classification task
            # convert raw predictions to class labels
            # add a CASE statement in order to avoid overflow errors
            score_to_class_query = "SELECT "
            for j in range(n_classes):
                score_to_class_query += "CASE WHEN -1.0*Score_{} > 500 THEN 0 ELSE 1.0/(1.0+EXP(-1.0*Score_{})) END AS PROB_{}, ".format(j, j, j)
            score_to_class_query = score_to_class_query[:-2] # remove the last ', '
            score_to_class_query += " FROM ({}) AS F".format(comb_query)

            if n_classes > 2:
                # find the greater prob
                case_stm = "CASE"
                for i in range(n_classes):
                    case_stm += " WHEN "
                    for j in range(n_classes):
                        if j == i:
                            continue
                        case_stm += "PROB_{} >= PROB_{} AND ".format(i, j)
                    case_stm = case_stm[:-5]  # remove the last ' AND '
                    case_stm += " THEN {}\n".format(classes[i])
                case_stm += "END AS Score"

                final_query = "SELECT {} FROM ({}) AS F".format(case_stm, score_to_class_query)
            else:
                final_query = "SELECT CASE WHEN PROB_0 > (1-PROB_0) THEN 1 ELSE 0 END AS Score FROM ({}) AS F".format(
                    score_to_class_query)
        else:
            final_query = comb_query

        return final_query

    def query(self, gbm: (GradientBoostingClassifier, GradientBoostingRegressor), features: list, table_name: str):
        """
        This method generates the SQL query that implements the GBM inference.

        :param gbm: the fitted Sklearn's Gradient Boosting model
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the GBM inference
        """

        error_msg = "Only GradientBoostingClassifier/GradientBoostingRegressor type is allowed fo param 'gbm'."
        assert isinstance(gbm, (GradientBoostingClassifier, GradientBoostingRegressor)), error_msg
        assert isinstance(features, list), "Only list data type is allowed for param 'features'"
        for f in features:
            assert isinstance(f, str)
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."

        # extract the parameters (i.e., the decision rules) from the fitted GBM
        gbm_params = GBMSQL.get_params(gbm, features, is_classification=self.classification, nested=self.nested,
                                       merge_ohe_features=self.merge_ohe_features)

        # create the SQL query that implements the GBM inference
        query = self._gbm_to_sql(gbm_params, table_name)

        return query
