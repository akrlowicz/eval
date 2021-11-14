import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.dummy import DummyClassifier
from mlxtend.evaluate import paired_ttest_5x2cv



def compare_classifiers(model_list, X_train, y_train):
    mean_avg = []
    training_times = []

    for model in model_list:
        print(model.__class__.__name__)
        print('Cross-validation')

        start_time = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=10)
        end_time = time.time()

        # print(f'Accuracy: {scores}')
        print(f'Mean accuracy: {scores.mean():.4f}')
        print(f'Std accuracy: {scores.std():.4f}')
        print('')

        mean_avg.append(scores.mean())
        training_times.append(end_time - start_time)

    plt.figure(figsize=(12, 10))
    indices = np.arange(len(mean_avg))
    plt.barh(indices, mean_avg, 0.2, label="score", color="navy")
    plt.barh(indices + 0.3, training_times / np.max(training_times), 0.2, label="training time", color="c")
    plt.yticks(())
    plt.legend(loc="best")

    for i, c in zip(indices, [m.__class__.__name__ for m in model_list]):
        plt.text(-0.3, i, c)

    plt.show()
    
    
def evaluate_model(model, X_train, y_train, X_test, y_test, target_names=None):
    
    print(f'\n-----------------------{model.__class__.__name__}-----------------------')
    
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print('Train')
    print(f'Accuracy: {accuracy_score(y_train, y_pred_train):.4f}')

    print('Test')
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    print('Confusion matrix')
    plot_confusion_matrix(model, X_test, y_test, normalize="true", display_labels=target_names)
    plt.grid(False) # sns override the grid
    plt.show()
   



def exhaustive_search(model, parameters, X_train, y_train, cv=5):

  
  cv = GridSearchCV(model,
                  param_grid = parameters,
                  cv = cv,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0)


  cv.fit(X_train, y_train)

  print("Best estimator found by grid search:")
  print(cv.best_estimator_)
    



def hyperparameter_tuning(model, X_train, y_train, param_grid,cv=5, n_iter=100):
  
  searched = RandomizedSearchCV(model,
                    param_distributions = param_grid,
                    n_iter = n_iter,
                    cv = cv,
                    scoring = 'accuracy',
                    n_jobs = -1,
                    error_score = 0.0)

  searched.fit(X_train, y_train)
  estimator = searched.best_estimator_
  print("Best classifier found by Randomized Search")
  print(estimator)


  cv = pd.DataFrame(searched.cv_results_)

  return estimator, cv




# significance threshold of α=0.05 for rejecting the null hypothesis
# that both algorithms perform equally well on the dataset
# if p > α, null hypothesis cannot be rejected
# if p < α, null hypothesis can be rejected, significant difference


def significance_test(model, X_train, y_train):

    dummy = DummyClassifier(strategy='most_frequent', random_state=123)
    dummy.fit(X_train, y_train)
    t, p = paired_ttest_5x2cv(estimator1=dummy,
                          estimator2=model,
                          X=X_train, y=y_train,
                          random_seed=123)
    print(f't statistic: {t:.3f}')
    print(f'p value: {p:.3f}')
