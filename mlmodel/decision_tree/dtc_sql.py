import numpy as np
from collections import Iterable
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor

class DTMSQL(object):
    """
    This class implements the SQL wrapper for a Sklearn Decision Tree Model (DTM).
    """

    def __init__(self, classification=False):
        """
        This method initializes the variables needed for the Gradient Boosting Model wrapper.

        :param classification: boolean flag that indicates whether the DTM is used in a classification or regression
                               task
        """

        if not isinstance(classification, bool):
            raise TypeError("Wrong data type for parameter classification. Only boolean data type is allowed.")

        self.classification = classification

    @staticmethod
    def get_dtc_rules(tree, feature_names):
        """
        This method extracts the rules from a "base or derived" Sklearn Decision Tree object.
        A single rule is composed by multiple conditions (expressed as tuples) and a final integer value:
         (parent_node_id, split (i.e., l for left/<= and r for right/>), parent_test_threshold, parent_test_features),
         (parent_node_id, split (i.e., l for left/<= and r for right/>), parent_test_threshold, parent_test_features),
         ...
         node_id <- leaf node
        The method outputs a flatten list containing the previous format repeated for each rule of the Decision Tree
        object.

        :param tree: "base or derived" Sklearn Decision Tree object
        :param feature_names: list of feature names
        :return: ("base or derived" Sklearn Decision Tree object object, "base or derived" Sklearn Decision Tree rules)
        """

        if not isinstance(tree, BaseDecisionTree):
            raise TypeError(
                "Wrong data type for parameter tree. Only Sklearn BaseDecisionTree data type is allowed.")

        if not isinstance(feature_names, Iterable):
            raise TypeError("Wrong data type for parameter feature_names. Only iterable data type is allowed.")

        for f in feature_names:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for feature_names elements. Only string data type is allowed.")

        decision_tree_rules = []

        left = tree.tree_.children_left  # left child for each node
        right = tree.tree_.children_right  # right child for each node
        threshold = tree.tree_.threshold  # test threshold for each node
        features = [feature_names[i] for i in tree.tree_.feature]  # names of the features used by the tree
        # (Sklearn provides the indexes of the feature)
        progress_rule = 0

        def recurse(left, right, child, lineage=None):

            # starting from the child node navigate the tree until the root is reached

            if lineage is None:  # if the current node is a leaf, save its id
                lineage = [child]
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
                return recurse(left, right, parent, lineage)

        # get ids of child nodes
        idx = np.argwhere(left == -1)[:, 0]
        # loop over child nodes
        for child in idx:
            # find the path that connects the current child node with the root of the tree
            for node in recurse(left, right, child):
                decision_tree_rules.append(node)
                progress_rule += 1

        return tree, decision_tree_rules

    @staticmethod
    def get_dtc_rules_by_level(tree, feature_names, weight=None):
        """
        This method extracts the rules from a "base or derived" Sklearn Decision Tree object and groups them by tree
        level. In more details, all the tests applied in the same tree level are grouped.
        Each test follows the format (feature, operator, threshold). A Python dictionary is return where the keys
        correspond to the tree levels and the values to the list of tests.

        :param tree: "base or derived" Sklearn Decision Tree object
        :param feature_names: list of feature names
        :param weight: (optional) parameter to weight the leaf scores
        :return: (rules grouped by level, tree left children, tree right children)
        """

        if not isinstance(tree, BaseDecisionTree):
            raise TypeError(
                "Wrong data type for parameter tree. Only Sklearn BaseDecisionTree data type is allowed.")

        if not isinstance(feature_names, Iterable):
            raise TypeError("Wrong data type for parameter feature_names. Only iterable data type is allowed.")

        for f in feature_names:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for feature_names elements. Only string data type is allowed.")

        if weight is not None:
            if not isinstance(weight, (int, float)):
                raise TypeError("Wrong data type for parameter weight. Only integer or float data type is allowed.")

        decision_tree_rules = {}

        # get for each node, left, right child nodes and threshold and features in their tests
        left = tree.tree_.children_left  # left child for each node
        right = tree.tree_.children_right  # right child for each node
        thresholds = tree.tree_.threshold  # test threshold for each node
        # features = [feature_names[i] for i in tree.tree_.feature]
        features = tree.tree_.feature  # indexes of the features used by the tree

        def visit_tree(node, parent, left, right, thresholds, features, level):

            cond = None

            # check if current node has a right child
            if right[node] != -1:
                if features[node] < 0:
                    raise Exception("Trying to retrieving the feature name from leaf node.")
                cond = (features[node], 'r', thresholds[node])  # , feature_names[features[node]])
                # visit right child
                visit_tree(right[node], node, left, right, thresholds, features, level + 1)

            # check if current node has a left child
            if left[node] != -1:
                if features[node] < 0:
                    raise Exception("Trying to retrieving the feature name from leaf node.")
                cond = (features[node], 'l', thresholds[node])  # , feature_names[features[node]])
                # visit left child
                visit_tree(left[node], node, left, right, thresholds, features, level + 1)

            # leaf node
            if left[node] == -1 and right[node] == -1:
                cond = tree.tree_.value[node][0][0]
                # if the user provided a weight, modify the leaf score
                if weight:
                    cond *= weight

            if level not in decision_tree_rules:
                decision_tree_rules[level] = [(node, parent, cond)]
            else:
                decision_tree_rules[level].append((node, parent, cond))

        # start tree visit from the root node
        root = 0
        visit_tree(root, -1, left, right, thresholds, features, 0)

        return decision_tree_rules, left, right

    @staticmethod
    def dtc_rules_to_sql(tree, decision_tree_rules, classification=False):
        """
        This method converts the rules extracted from a "base or derived" Sklearn Decision Tree object into SQL case
        statements.

        :param tree: "base or derived" Sklearn Decision Tree object
        :param decision_tree_rules: rules extracted from a "base or derived" Sklearn Decision Tree object with a format
                                    compliant with the one provided by the get_dtc_rules function
        :param classification: boolean flag that indicates whether the DTM is used in a classification or regression
                       task
        :return: list of SQL representations for each tree rule
        """

        if not isinstance(tree, BaseDecisionTree):
            raise TypeError(
                "Wrong data type for parameter tree. Only Sklearn BaseDecisionTree data type is allowed.")

        if not isinstance(decision_tree_rules, Iterable):
            raise TypeError("Wrong data type for parameter decision_tree_rules. Only iterable data type is allowed.")

        for cond in decision_tree_rules:
            if not isinstance(cond, (tuple, int, np.integer)):
                raise TypeError(
                    "Wrong data type for decision_tree_rules conditions. Only tuple or integer data type is allowed.")

            if isinstance(cond, tuple):
                if len(cond) != 4:
                    raise TypeError("Wrong data format for decision_tree_rules condition. Wrong length.")

                if not isinstance(cond[0], (int, np.integer)):
                    raise TypeError(
                        "Wrong data format for decision_tree_rules condition. Wrong parent node id data type.")

                if not isinstance(cond[1], str):
                    raise TypeError(
                        "Wrong data format for decision_tree_rules condition. Wrong split data type.")

                if not isinstance(cond[2], (float, str)):
                    raise TypeError(
                        "Wrong data format for decision_tree_rules condition. Wrong parent test threshold data type.")

                if not isinstance(cond[3], str):
                    raise TypeError(
                        "Wrong data format for decision_tree_rules condition. Wrong parent test feature data type.")
            else:
                if not isinstance(cond, (int, np.integer)):
                    raise TypeError(
                        "Wrong data format for decision_tree_rules condition. Wrong leaf node id data type.")

        if not isinstance(classification, bool):
            raise TypeError("Wrong data type for parameter classification. Only boolean data type is allowed.")

        rules_sql = []
        rule_string = "CASE WHEN "
        # loop over the rules
        for item in decision_tree_rules:

            if not isinstance(item, tuple):  # the item is the index of a leaf node
                rule_string = rule_string[:-5]  # remove 'WHEN '
                if classification: # classification task
                    predicted_class = np.argmax(tree.tree_.value[item][0]) # get leaf majority class
                else: # regression task
                    predicted_class = tree.tree_.value[item][0][0]  # get the leaf predicted score
                rule_string += " THEN {}".format(predicted_class)
                rules_sql.append(rule_string)
                rule_string = "WHEN "
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
                rule_string += "{} {} {} and ".format(feature_name, op, thr)

        return rules_sql

    @staticmethod
    def get_params(dtm, features, classification=False, split_rules_by_tree_level=False):
        """
        This method extracts from the Sklearn Decision Tree Model all the fitted parameters (i.e., tree rules) needed to
        replicate in SQL the inference. The extracted rules can be returned in multiple formats: flat list or grouped by
        tree level.

        :param dtm: "base or derived" Sklearn Decision Tree object
        :param features: the list of features
        :param classification: boolean flag that indicates whether the DTM is used in a classification or regression
                               task
        :param split_rules_by_tree_level: (optional) boolean flag that indicates whether the extracted rules have to be
                                          grouped by tree level
        :return: Python dictionary containing the DTM fitted parameters (i.e., tree rules)
        """

        if not isinstance(dtm, (DecisionTreeClassifier, DecisionTreeRegressor)):
            error_msg = "Wrong data type for parameter gbm. "
            error_msg += "Only the Sklearn DecisionTreeClassifier/DecisionTreeRegressor type is allowed."
            raise TypeError(error_msg)

        if not isinstance(features, Iterable):
            raise TypeError("Wrong data type for parameter features. Only iterable data type is allowed.")

        for f in features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single features. Only string data type is allowed.")

        if not isinstance(classification, bool):
            raise TypeError("Wrong data type for parameter classification. Only boolean data type is allowed.")

        if not isinstance(split_rules_by_tree_level, bool):
            raise TypeError(
                "Wrong data type for parameter split_rules_by_tree_level. Only boolean data type is allowed.")

        # extract the decision rules from the Decision Tree Model
        if not split_rules_by_tree_level:
            # extract the rules from the tree
            estimator, decision_tree_rules = DTMSQL.get_dtc_rules(dtm, features)
            # convert the rules in a SQL format
            rules_strings = DTMSQL.dtc_rules_to_sql(dtm, decision_tree_rules, classification)
            # save the rules, the SQL rules and the decision tree model
            tree_params = {"estimator": estimator, "string_rules": ' '.join(rules_strings),
                           "weight": 1, "rules": decision_tree_rules}
        else:
            # get the tree rules grouped by tree levels
            decision_tree_rules, left, right = DTMSQL.get_dtc_rules_by_level(dtm, features, 1)

            # save tree parameters
            tree_params = {"estimator": dtm, "string_rules": None, "weight": 1,
                           "rules": decision_tree_rules, "left_nodes": left, "right_nodes": right}

        return tree_params

    def _dtm_to_sql(self, tree_params, table_name):
        """
        This method generates the SQL query that implements the DTM inference according to multiple approach (i.e.,
        rule-based approach where a CASE statement for each tree rule is created, level-based approach where a CASE
        statement for each level of tree structure is generated).

        :param tree_params: the parameters extracted from the Sklearn's DTM object
        :param table_name: the name of the table from which the SQL query will read
        :return: the SQL query that implements the GBM inference
        """

        if not isinstance(tree_params, dict):
            raise TypeError("Wrong data type for parameter tree_params. Only Python dictionary data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        params_keys = ["estimator", "string_rules", "weight", "rules"]
        if not all(p in params_keys for p in tree_params):
            raise ValueError("Wrong data format for parameter tree_params.")

        query = "SELECT "
        # create a CASE statement for the tree
        query += "{} END AS Score".format(tree_params["string_rules"])

        query += " FROM {}".format(table_name)

        return query

    def query(self, dtm, features, table_name):
        """
        This method generates the SQL query that implements the GBM inference.

        :param dtm: the fitted Sklearn Decision Tree model
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :return: the SQL query that implements the GBM inference
        """

        if not isinstance(dtm, (DecisionTreeClassifier, DecisionTreeRegressor)):
            error_msg = "Wrong data type for parameter gbm. "
            error_msg += "Only the Sklearn DecisionTreeClassifier/DecisionTreeRegressor type is allowed."
            raise TypeError(error_msg)

        if not isinstance(features, Iterable):
            raise TypeError("Wrong data type for parameter features. Only iterable data type is allowed.")

        for f in features:
            if not isinstance(f, str):
                raise TypeError("Wrong data type for single features. Only string data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        # extract the parameters (i.e., the decision rules) from the fitted DTM
        tree_parameters = DTMSQL.get_params(dtm, features, classification=self.classification)

        # create the SQL query that implements the DTM inference
        query = self._dtm_to_sql(tree_parameters, table_name)

        return query
