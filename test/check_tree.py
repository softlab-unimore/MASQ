import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt


def check_path(clf: (DecisionTreeRegressor, DecisionTreeClassifier), data: np.ndarray, sample_id: int, plot=False,
               save_fig=None):
    assert isinstance(clf, (DecisionTreeRegressor, DecisionTreeClassifier))
    assert isinstance(data, np.ndarray)
    assert isinstance(sample_id, int)
    assert sample_id < data.shape[0]
    assert isinstance(plot, bool)
    if save_fig is not None:
        assert isinstance(save_fig, str)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(data)
    leaf_id = clf.apply(data)

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample {id}:\n'.format(id=sample_id))
    last = None
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        last = node_id
        # check if value of the split feature for sample 0 is below threshold
        if (data[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
              "{inequality} {threshold})".format(
            node=node_id,
            sample=sample_id,
            feature=feature[node_id],
            value=data[sample_id, feature[node_id]],
            inequality=threshold_sign,
            threshold=threshold[node_id]))
    print(last)
    r = children_right[last]
    l = children_left[last]
    print(clf.tree_.value[r][0])
    print(clf.tree_.value[l][0])

    print(clf.predict([data[sample_id, :]]))

    if plot:
        plot_tree(clf)
        if save_fig is not None:
            plt.savefig(save_fig, bbox_inches="tight")
        else:
            plt.show()
