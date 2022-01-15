import numpy as np
from matplotlib import pyplot as plt
import time

from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_tables

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

from prettytable import PrettyTable

def fitted_clf_evaluation(fitted_model, target_names, X_test, y_test):

    f, axes = plt.subplots(1,2,figsize=(12,4.8))
    f.suptitle(fitted_model.__class__.__name__)

    #Classification Report
    ax1 = ClassificationReport(fitted_model, classes=target_names, support=True, ax=axes[0])
    ax1.score(X_test, y_test)  
    ax1.finalize()

    #Confusion Matrix
    ax2 = ConfusionMatrix(fitted_model, classes=target_names, percent=True, ax=axes[1])
    ax2.score(X_test, y_test)  # Evaluate the model on the test data
    ax2.ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(rotation=45, horizontalalignment="right")
    ax2.finalize()

    plt.show()

def model_compare(key, model_list):

    two_models = key.split(' vs ')
    first_model = int(two_models[0].split('_')[-1])
    second_model = int(two_models[1].split('_')[-1])

    return model_list[first_model], model_list[second_model]

def multiple_mcnemar(model_name_list, preds_array, y, sig_level):

    # creates the 'correct' value for use in the McNemar table
    y_true = np.array([1] * len(y))

    converted_pred_array = []

    for i in range(len(preds_array)):
        converted_pred_array.append((preds_array[i]==y).astype(int))

    q, p_value = cochrans_q(y_true,*converted_pred_array)

    significance = p_value < sig_level

    scientific_notation="{:.2e}".format(p_value)

    print(f"-----Cochran's Q Test-----")
    print(f'============================')
    print(f'Q-Score         {q:.4f}')
    print(f'p-value         {scientific_notation}')
    print(f'Reject? ({1-sig_level:.0%})      {significance}')
    print('\n')

    mctable = mcnemar_tables(y_true, *converted_pred_array)

    pairwise_table = PrettyTable()
    pairwise_table.field_names = ['Model 1', 'Model 2', 'Chi²', 'p-Value', 'Reject?']

    for key, value in mctable.items():
        chi2, p = mcnemar(ary=value, corrected=True)
        first_model, second_model = model_compare(key, model_name_list)
        reject_null = p < sig_level

        pairwise_table.add_row([first_model, second_model, chi2, p, reject_null])

    print(pairwise_table)

def timed_fit(model, X, y):

  # run a single fit for each model and get training time
  start_time = time.time()
  model.fit(X, y)
  end_time = time.time()

  fitting_time = end_time - start_time

  return model, fitting_time
