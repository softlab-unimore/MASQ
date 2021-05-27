from sklearn.tree import BaseDecisionTree, export_graphviz
import os
import graphviz


def plot_tree(dtm: BaseDecisionTree, features: list, file_name: str, class_labels: list=None):
    """
    This method plots the structure of a BaseDecisionTree object and saves it in a file.

    :param dtm: BaseDecisionTree object
    :param features: the list of features
    :param file_name: name of the file where to save the plot
    :param class_labels: (optional) class values to insert in the plot
    :return: None
    """

    assert isinstance(dtm, BaseDecisionTree), "Only BaseDecisionTree data type is allowed for param 'dtm'."
    assert isinstance(features, list), "Only list data type is allowed for param 'features'."
    assert isinstance(file_name, str), "Only string data type is allowed for param 'file_name'."
    assert os.path.exists(os.sep.join(file_name.split(os.sep)[:-1]))
    if class_labels is not None:
        assert isinstance(class_labels, list), "Only list data type is allowed for param 'class_labels'."

    # create plot data
    if not class_labels:
        dot_data = export_graphviz(dtm, out_file=None, feature_names=features, filled=True,
                                   rounded=True, special_characters=True)
    else:

        dot_data = export_graphviz(dtm, out_file=None, feature_names=features, class_names=class_labels,
                                   filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    # save plot into file
    graph.render(file_name)
