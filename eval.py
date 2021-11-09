import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_score, classification_report
from sklearn.metrics import accuracy_score


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
    
    
def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    print(f'\n-----------------------{model.__class__.__name__}-----------------------')
    
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print('Train')
    print(f'Accuracy: {accuracy_score(y_train, y_pred_train):.4f}')

    print('Test')
    print(classification_report(y_test, y_pred_test, target_names=['Persistent', 'Non-Persistent']))

    print('Confusion matrix')
    plot_confusion_matrix(model, X_test, y_test, normalize="true", display_labels=['Persistent', 'Non-Persistent'])
    plt.grid(False) # sns override the grid
    plt.show()
    
    