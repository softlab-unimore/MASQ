import numpy as np
from sklearn.tree import BaseDecisionTree
from utils.dbms_utils import DBMSUtils


class DTMSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn's Decision Tree Model (DTM).
    """

    def __init__(self, classification: bool = False):
        """
        This method initializes the state of the Decision Tree Model SQL wrapper.

        :param classification: boolean flag that indicates whether the DTM is used in classification or regression
        """

        assert isinstance(classification, bool), "Only boolean data type is allowed for param 'classification'."

        self.classification = classification
        self.nested = True
        self.merge_ohe_features = None
        self.dbms = None
        self.mode = None

    def set_mode(self, mode: str):
        assert isinstance(mode, str), "Wrong data type for param 'mode'."
        self.mode = mode

    def set_dbms(self, dbms: str):
        self.dbms = dbms

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
    def get_flat_rules(tree: BaseDecisionTree, feature_names: list, is_classification: bool):
        """
        This method extracts the tree decision paths from a BaseDecisionTree object.
        Each decision path is composed of multiple conditions (extracted from the tree internal nodes) and a final score
        (extracted from the tree leaf). The following format is used to represent a single decision path:
         (parent_node_id, split (i.e., l for left/<= and r for right/>), parent_test_threshold, parent_test_features),
         (parent_node_id, split (i.e., l for left/<= and r for right/>), parent_test_threshold, parent_test_features),
         ...
         score

        :param tree: BaseDecisionTree object
        :param feature_names: list of feature names
        :param is_classification: boolean flag that indicates whether the DTM is used in classification or regression
        :return: list containing the tree decision paths
        """

        assert isinstance(tree, BaseDecisionTree), "Only BaseDecisionTree data type is allowed for param 'tree'."
        assert isinstance(feature_names, list), "Only list data type is allowed for param 'features_names'."
        for f in feature_names:
            assert isinstance(f, str)
        assert isinstance(is_classification, bool), "Only bool data type is allowed for param 'is_classification'."

        decision_tree_rules = []

        left = tree.tree_.children_left  # left child for each node
        right = tree.tree_.children_right  # right child for each node
        threshold = tree.tree_.threshold  # test threshold for each node
        features = [f"{feature_names[i]}" for i in tree.tree_.feature]  # names of the features used by the tree
        scores = tree.tree_.value
        if is_classification:
            classes = tree.classes_

        def deep_visit_tree(left, right, child, lineage=None):

            # starting from the child node navigate the tree until the root is reached

            if lineage is None:  # if the current node is a leaf, save its id
                if is_classification:
                    lineage = [classes[np.argmax(scores[child][0])]]
                else:
                    lineage = [tree.tree_.value[child][0][0]]
            if child in left:  # if the current node is a left child get the parent and save the left direction
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:  # if the current node is a right child get the parent and save the right direction
                parent = np.where(right == child)[0].item()
                split = 'r'

            # save the info about how the current node is reachable by its parent
            lineage.append((parent, split, threshold[parent], features[parent]))

            if parent == 0:  # if the parent of the current node is the root of the tree return the reversed path
                lineage.reverse()
                return lineage
            else:  # if the parent of the current node is not the root of the tree continue the path navigation
                return deep_visit_tree(left, right, parent, lineage)

        # get ids of child nodes
        idx = np.argwhere(left == -1)[:, 0]
        if len(idx) == 1:   # if the tree is composed of a single leaf
            if is_classification:
                return [classes[np.argmax(scores[0][0])]]
            else:
                return [tree.tree_.value[0][0][0]]
        # loop over child nodes
        for child in idx:
            # find the path that connects the current child node with the root of the tree
            for node in deep_visit_tree(left, right, child):
                decision_tree_rules.append(node)

        return decision_tree_rules

    @staticmethod
    def _check_rule_format(rules: list):
        """
        This method checks the format of the input rules. The right format is the one adopted by the 'get_flat_rules'
        method.

        :param rules: rules extracted from a BaseDecisionTree object to check
        """
        assert isinstance(rules, list), "Only list data type is allowed for param 'rules'."
        error_msg = "Wrong data format for param 'rules'."
        for cond in rules:
            assert isinstance(cond, (tuple, int, np.integer, np.float, float)), error_msg
            if isinstance(cond, tuple):
                assert len(cond) == 4, error_msg
                assert isinstance(cond[0], (int, np.integer)), error_msg
                assert isinstance(cond[1], str), error_msg
                assert isinstance(cond[2], (float, str)), error_msg
                assert isinstance(cond[3], str), error_msg
            else:
                assert isinstance(cond, (int, np.integer, np.float, float)), error_msg

    @staticmethod
    def convert_rules_to_sql(rules: list, dbms: str, merge_ohe_features: dict = None):
        """
        This method converts the rules extracted from a BaseDecisionTree object into SQL case statements.

        :param rules: rules extracted from a BaseDecisionTree object
        :param dbms: the name of the dbms
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: string containing the SQL query
        """

        DTMSQL._check_rule_format(rules)
        if merge_ohe_features is not None:
            DTMSQL._check_merge_ohe_features(merge_ohe_features)

        dbms_util = DBMSUtils()

        sql_query = ""
        sql_string = " CASE WHEN "
        # loop over the rules
        for item in rules:

            if not isinstance(item, tuple):  # the item is a leaf score
                sql_string = sql_string[:-5]  # remove 'WHEN '
                if sql_string == ' CASE ':  # case a tree is composed of only a leaf
                    sql_query += str(item)
                else:
                    sql_string += " THEN {} ".format(item)
                    sql_query += sql_string
                sql_string = "WHEN "
            else:  # the item is a rule condition
                op = item[1]
                thr = item[2]
                if op == 'l':
                    op = '<='
                elif op == 'r':
                    op = '>'
                else:  # when op is equals to '=' or '<>' the thr is a string
                    thr = "'{}'".format(thr)
                feature_name = item[3]

                if merge_ohe_features is not None:
                    # if ohe features have to be merged in the decision tree, the tree conditions are changed
                    #   feature_after_ohe > 0.5 becomes original_cat_feature = val
                    #   feature_after_ohe <= 0.5 becomes original_cat_feature <> val
                    if feature_name in merge_ohe_features:  # only categorical features should pass this test
                        mof = merge_ohe_features[feature_name]
                        feature_name = mof['feature_before_ohe']
                        old_op = op[:]
                        op = '='
                        if old_op == '<=':
                            if 0 <= thr:
                                op = '<>'
                        elif old_op == '>':
                            if 0 > thr:
                                op = '<>'
                        else:
                            raise ValueError("Wrong op.")

                        thr = "'{}'".format(mof['value'])

                feature_name = dbms_util.get_delimited_col(dbms, feature_name)
                sql_string += "{} {} {} and ".format(feature_name, op, thr)

        if 'CASE' in sql_query:     # ignore the case where a tree is composed of only a leaf
            sql_query += "END "

        return sql_query

    @staticmethod
    def get_sql_flat_rules(tree: BaseDecisionTree, feature_names: list, is_classification: bool, dbms: str,
                           merge_ohe_features: dict = None):
        """
        This method extracts the rules from a BaseDecisionTree object and convert them in SQL.

        :param tree: BaseDecisionTree object
        :param feature_names: list of feature names
        :param is_classification: boolean flag that indicates whether the DTM is used in classification or regression
        :param dbms: the name of the dbms
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: string containing the SQL query
        """

        if merge_ohe_features is not None:
            DTMSQL._check_merge_ohe_features(merge_ohe_features)

        rules = DTMSQL.get_flat_rules(tree, feature_names, is_classification)
        sql_query = DTMSQL.convert_rules_to_sql(rules, dbms, merge_ohe_features)

        return sql_query

    @staticmethod
    def get_sql_nested_rules(tree: BaseDecisionTree, feature_names: list, is_classification: bool, dbms: str,
                             merge_ohe_features: dict = None):
        """
        This method extracts the rules from a BaseDecisionTree object and convert them in SQL.

        :param tree: BaseDecisionTree object
        :param feature_names: list of feature names
        :param is_classification: boolean flag that indicates whether the DTM is used in classification or regression
        :param dbms: the name of the dbms
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: string containing the SQL query
        """

        assert isinstance(tree, BaseDecisionTree), "Only BaseDecisionTree data type is allowed for param 'tree'."
        assert isinstance(feature_names, list), "Only list data type is allowed for param 'features_names'."
        for f in feature_names:
            assert isinstance(f, str)
        assert isinstance(is_classification, bool), "Only bool data type is allowed for param 'is_classification'."
        if merge_ohe_features is not None:
            DTMSQL._check_merge_ohe_features(merge_ohe_features)

        dbms_util = DBMSUtils()

        # get for each node, left, right child nodes, thresholds and features
        left = tree.tree_.children_left  # left child for each node
        right = tree.tree_.children_right  # right child for each node
        thresholds = tree.tree_.threshold  # test threshold for each node
        features = [feature_names[i] for i in tree.tree_.feature]
        # features = tree.tree_.feature  # indexes of the features used by the tree
        if is_classification:
            classes = tree.classes_

        def visit_tree(node):

            # leaf node
            if left[node] == -1 and right[node] == -1:
                if is_classification:
                    return " {} ".format(classes[np.argmax(tree.tree_.value[node][0])])
                else:
                    return " {} ".format(tree.tree_.value[node][0][0])

            # internal node
            op = '<='
            feature = features[node]
            thr = thresholds[node]

            if merge_ohe_features is not None:
                # if ohe features have to be merged in the decision tree, the tree conditions are changed
                #   feature_after_ohe > 0.5 becomes original_cat_feature = val
                #   feature_after_ohe <= 0.5 becomes original_cat_feature <> val
                if feature in merge_ohe_features:                      # only categorical features should pass this test
                    mof = merge_ohe_features[feature]
                    feature = mof['feature_before_ohe']
                    thr = "'{}'".format(mof['value'])
                    op = '<>'

            sql_dtm_rule = f" CASE WHEN {dbms_util.get_delimited_col(dbms, feature)} {op} {thr} THEN"

            # check if current node has a left child
            if left[node] != -1:
                sql_dtm_rule += visit_tree(left[node])

            sql_dtm_rule += "ELSE"

            # check if current node has a right child
            if right[node] != -1:
                sql_dtm_rule += visit_tree(right[node])

            sql_dtm_rule += "END "

            return sql_dtm_rule

        # start tree visit from the root node
        root = 0
        sql_dtm_rules = visit_tree(root)

        return sql_dtm_rules

    @staticmethod
    def get_params(dtm: BaseDecisionTree, features: list, is_classification: bool, nested: bool, dbms: str,
                   merge_ohe_features: dict = None):
        """
        This method extracts the tree rules from the Sklearn's Decision Tree Model and creates their SQL representation.

        :param dtm: BaseDecisionTree object
        :param features: the list of features
        :param is_classification: boolean flag that indicates whether the DTM is used in classification or regression
        :param nested: boolean flag that indicates whether to use the nested SQL conversion technique
        :param dbms: the name of the dbms
        :param merge_ohe_features: (optional) ohe feature map to be merged in the decision rules
        :return: Python dictionary containing the parameters extracted from the fitted DTM
        """

        assert isinstance(dtm, BaseDecisionTree), "Only BaseDecisionTree type is allowed fo param 'dtm'."
        assert isinstance(features, list), "Only list data type is allowed for param 'features'"
        for f in features:
            assert isinstance(f, str)
        assert isinstance(is_classification, bool), "Only boolean data type is allowed for param 'is_classification'."
        assert isinstance(nested, bool)
        if merge_ohe_features is not None:
            DTMSQL._check_merge_ohe_features(merge_ohe_features)

        # extract the decision rules from the Decision Tree Model
        if nested:
            sql_rules = DTMSQL.get_sql_nested_rules(dtm, features, is_classification, dbms,
                                                    merge_ohe_features=merge_ohe_features)
        else:
            sql_rules = DTMSQL.get_sql_flat_rules(dtm, features, is_classification, dbms,
                                                  merge_ohe_features=merge_ohe_features)
        tree_params = {"estimator": dtm, "sql_rules": sql_rules}

        return tree_params

    def _dtm_to_sql(self, tree_params: dict, table_name: str):
        """
        This method generates the SQL query that implements the DTM inference.

        :param tree_params: the parameters extracted from the Sklearn's DTM object
        :param table_name: the name of the table from which the SQL query will read
        :return: the SQL query that implements the DTM inference
        """

        assert isinstance(tree_params, dict), "Only Python dictionary data type is allowed for param 'tree_params'."
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."
        params_keys = ["estimator", "sql_rules"]
        assert all(p in params_keys for p in tree_params), "Wrong data format for parameter 'tree_params'."

        query = "SELECT{} AS Score".format(tree_params["sql_rules"])
        query += " FROM {}".format(table_name)

        return query

    def query(self, dtm: BaseDecisionTree, features: list, table_name: str):
        """
        This method generates the SQL query that implements the DTM inference.

        :param dtm: the fitted Sklearn Decision Tree model
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: string that contains the SQL query that implements the DTM inference
        """

        assert isinstance(dtm, BaseDecisionTree), "Only the BaseDecisionTree type is allowed fo param 'dtm'."
        assert isinstance(features, list), "Only list data type is allowed for param 'features'"
        for f in features:
            assert isinstance(f, str)
        assert isinstance(table_name, str), "Only string data type is allowed for param 'table_name'."

        assert self.dbms is not None, "No dbms selected."

        # extract the parameters (i.e., the decision rules) from the fitted DTM
        dtm_params = DTMSQL.get_params(dtm, features, is_classification=self.classification, nested=self.nested,
                                       dbms=self.dbms, merge_ohe_features=self.merge_ohe_features)

        # create the SQL query that implements the DTM inference
        pre_inference_query = None
        query = self._dtm_to_sql(dtm_params, table_name)

        return pre_inference_query, query
