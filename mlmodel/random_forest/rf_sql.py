import numpy as np
from collections import Iterable
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RFMSQL(object):
    """
    This class implements the SQL wrapper for a Random Forest Model (RFM).
    """

    def __init__(self, classification=False):
        """
        This method initializes the variables needed for the Random Forest Model wrapper.

        :param classification: boolean flag that indicates whether the RFM is used in a classification or regression
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
        :param classification: boolean flag that indicates whether the RFM is used in a classification or regression
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
    def get_params(rfm, features, classification=False, split_rules_by_tree_level=False):
        """
        This method extracts from the Sklearn RFM all the fitted parameters (i.e., tree rules) needed to replicate in
        SQL the RFM inference. The extracted rules can be returned in multiple formats: flat list or grouped by tree
        level.

        :param rfm: the fitted Sklearn Random Forest model
        :param features: the list of features
        :param classification: boolean flag that indicates whether the RFM is used in a classification or regression
                               task
        :param split_rules_by_tree_level: boolean flag that indicates whether the extracted rules have to be grouped by
                                          tree level
        :return: Python dictionary containing the RFM fitted parameters (i.e., tree rules)
        """

        if not isinstance(rfm, (RandomForestClassifier, RandomForestRegressor)):
            error_msg = "Wrong data type for parameter gbm. "
            error_msg += "Only Sklearn RandomForestClassifier/RandomForestRegressor data type is allowed."
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

        # extract decision rules from the RFM decision trees
        # get trees
        trees = rfm.estimators_
        trees_parameters = []
        # loop over trees
        for index, tree in enumerate(trees):

            if not split_rules_by_tree_level:
                # extract the rules from the current tree
                estimator, decision_tree_rules = RFMSQL.get_dtc_rules(tree, features)
                # convert the rules in a SQL format
                rules_strings = RFMSQL.dtc_rules_to_sql(tree, decision_tree_rules, classification)
                # save the rules, the SQL rules, the learning rate and the model related to the current tree
                tree_params = {"estimator": estimator, "string_rules": ' '.join(rules_strings),
                               "weight": 1.0/len(trees), "rules": decision_tree_rules}
            else:
                # get tree rules grouped by tree levels
                decision_tree_rules, left, right = RFMSQL.get_dtc_rules_by_level(tree, features, 1.0/len(trees))

                # save tree parameters
                tree_params = {"estimator": tree, "string_rules": None, "weight": 1.0/len(trees),
                               "rules": decision_tree_rules, "left_nodes": left, "right_nodes": right}
            trees_parameters.append(tree_params)

        return trees_parameters

    @staticmethod
    def rfm_to_sql(trees_params, table_name, classification=False, class_labels=None):
        """
        This method generates the SQL query that implements the RFM inference according to multiple approach (i.e.,
        rule-based approach where a CASE statement for each tree rule is created, level-based approach where a CASE
        statement for each level of tree structure (and among all the the GBM trees in parallel) is generated).

        :param trees_params: the parameters extracted from the Sklearn's GBM object
        :param table_name: the name of the table from which the SQL query will read
        :param classification: boolean flag that indicates whether the RFM is used in a classification or regression
                               task
        :param class_labels: (optional) the labels of the class attribute
        :return: the SQL query that implements the GBM inference
        """

        if not isinstance(trees_params, Iterable):
            raise TypeError("Wrong data type for parameter trees_parmas. Only iterable data type is allowed.")

        for tree_params in trees_params:
            if not isinstance(tree_params, dict):
                raise TypeError(
                    "Wrong data type for trees_params elements. Only Python dictionary data type is allowed.")

        if not isinstance(table_name, str):
            raise TypeError("Wrong data type for parameter table_name. Only string data type is allowed.")

        if not isinstance(classification, bool):
            raise TypeError("Wrong data type for parameter classification. Only boolean data type is allowed.")

        if class_labels is not None:
            if not isinstance(class_labels, Iterable):
                raise TypeError("Wrong data type for parameter class_labels. Only iterable data type is allowed.")

        params_keys = ["estimator", "string_rules", "weight", "rules"]
        for tree_params in trees_params:
            if not all(p in params_keys for p in tree_params):
                raise ValueError("Wrong data format for parameter trees_params.")

        if classification and class_labels is None:
            raise ValueError("No class labels provided in classification task.")

        query = "SELECT "
        # loop over the trees
        # create a CASE statement for each tree
        for i in range(len(trees_params)):
            tree_params = trees_params[i]
            sql_case = tree_params["string_rules"]

            sql_case += " END AS tree_{},".format(i)

            query += sql_case

        query = query[:-1] # remove the last ","

        query += " FROM {}".format(table_name)

        if classification: # classification task

            class_labels = list(class_labels)

            # find the majority class

            # count the number of trees that have predicted the same class label
            majority_class_query = "SELECT "
            for class_ix, class_label in enumerate(class_labels):
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
            for i in range(len(class_labels)):
                case_stm += " WHEN "
                for j in range(len(class_labels)):
                    if j == i:
                        continue
                    case_stm += "class_{} > class_{} AND ".format(i, j)
                case_stm = case_stm[:-5]  # remove the last ' AND '
                case_stm += " THEN {}\n".format(class_labels[i])
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

    def query(self, rfm, features, table_name, class_labels=None):
        """
        This method generates the SQL query that implements the RFM inference.

        :param rfm: the fitted Sklearn Random Forest model
        :param features: the list of features
        :param table_name: the name of the table or the subquery where to read the data
        :param class_labels: (optional) the labels of the class attribute
        :return: the SQL query that implements the GBM inference
        """

        if not isinstance(rfm, (RandomForestClassifier, RandomForestRegressor)):
            error_msg = "Wrong data type for parameter gbm. "
            error_msg += "Only Sklearn RandomForestClassifier/RandomForestRegressor data type is allowed."
            raise TypeError(error_msg)

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

        # extract the parameters (i.e., the decision rules) from the fitted GBM
        trees_parameters = RFMSQL.get_params(rfm, features, self.classification)

        if self.classification:
            if class_labels is None:

                try:
                    n_classes = rfm.n_classes_
                except AttributeError:
                    raise AttributeError(
                        "No attribute n_classes_ found in the Random Forest Model. Is it a classifier?")

                class_labels = range(n_classes)

        # create the SQL query that implements the GBM inference
        query = RFMSQL.rfm_to_sql(trees_parameters, table_name, self.classification, class_labels=class_labels)

        return query
