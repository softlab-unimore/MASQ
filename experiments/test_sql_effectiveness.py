from workflow.utils import get_sql_predictions_effectiveness

if __name__ == '__main__':
    sklearn_preds_file = "../data/result/test_run_dbFalse.csv"
    sql_preds_file = "../data/result/test_run_dbTrue.csv"

    res = get_sql_predictions_effectiveness(sklearn_preds_file, sql_preds_file)
    print(res)