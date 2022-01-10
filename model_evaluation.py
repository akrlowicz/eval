import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix


def run_cv(model, X_train, y_train, name=None):

  # run 5-fold CV for each model and get training time
  start_time = time.time()
  scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
  end_time = time.time()

  training_time = end_time - start_time

  # individual model print-out
  if name is None:
    print(model.__class__.__name__)
  else:
    print(name)
  print(f'Mean weighted F1-score: {scores.mean():.4f}')
  print(f'Std deviation of F1-scores: {scores.std():.4f}')
  print(f'Time for 5 Cross-Validations: {training_time:,.4f} seconds')

  return scores, training_time


def holdout_evaluation(model, target_names, X_train, y_train, X_test, y_test):

    f, axes = plt.subplots(1,2,figsize=(12,4.8))
    f.suptitle(model.__class__.__name__)

    #Classification Report
    ax1 = ClassificationReport(model, classes=target_names, support=True, ax=axes[0])
    ax1.fit(X_train, y_train)  # Fit the visualizer and the model
    ax1.score(X_test, y_test)  # Evaluate the model on the test data
    ax1.finalize()

    #Confusion Matrix
    ax2 = ConfusionMatrix(model, classes=target_names, percent=True, ax=axes[1])
    ax2.fit(X_train, y_train)  # Fit the visualizer and the model
    ax2.score(X_test, y_test)  # Evaluate the model on the test data
    ax2.ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(rotation=45, horizontalalignment="right")
    ax2.finalize()

    plt.show()

def multi_cv_comparison(model_array, filename=None):
     
    mean_avg = []
    training_times = []
    score_df = pd.DataFrame({'score':[], 'model':[]})

    model_list_with_dummy = model_array.copy()

    for i in range(len(model_list_with_dummy)):

      model_name = model_list_with_dummy.iloc[i]['name']
      model = model_list_with_dummy.iloc[i]['model']
      X_train = model_list_with_dummy.iloc[i]['Xtrain']
      y_train = model_list_with_dummy.iloc[i]['ytrain']

      current_model_score, current_ttime = run_cv(model, X_train, y_train, model_name)

      mean_avg.append(current_model_score.mean())
      training_times.append(current_ttime)

      # build up dataframe in order to conduct ANOVA and Tukey
      model_name_series = pd.Series(model_name)
      score_series = pd.Series(current_model_score)
      model_series = model_name_series.repeat(len(score_series)).reset_index(drop=True)

      appended_df = pd.concat([model_series, score_series], axis =1, ignore_index = True)
      appended_df = appended_df.set_axis(["model", "score"], axis=1)
      
      score_df = score_df.append(appended_df, ignore_index=True)
    
    #plot comparison of accuracies and training times
    print('\n\n')
    plt.figure(figsize=(12, 4.8))
    indices = np.arange(len(mean_avg))
    plt.barh(indices, mean_avg, 0.2, label="score", color="navy")
    plt.barh(indices + 0.3, training_times / np.max(training_times), 0.2, label="training time", color="c")
    plt.gca().invert_yaxis()
    plt.yticks(())
    plt.legend(loc="best")

    for i, c in zip(indices, [model_list_with_dummy.iloc[m]['name'] for m in range(len(model_list_with_dummy))]):
        plt.text(-0.3, i, c)

    if filename is not None:
        plt.savefig(filename)
  
    plt.show()

    #ANOVA test
    F, pval = f_oneway(*(score_df[score_df['model'] == model_list_with_dummy.iloc[m]['name']]['score'] for m in range(len(model_list_with_dummy))))

    scientific_notation="{:.2e}".format(pval)

    print('\n\n')
    print(f'----One-Way ANOVA Test----')
    print(f'==========================')
    print(f'F-Score       {F:.4f}')
    print(f'p-value       {scientific_notation}')
    print('\n\n')

    #Tukey test at 99% significance level
    tukey = pairwise_tukeyhsd(endog=score_df['score'],
                          groups=score_df['model'],
                          alpha=0.01)

    print(tukey)
    print('\n\n')
