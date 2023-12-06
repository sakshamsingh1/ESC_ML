import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

path = 'data/esc50_features.csv'
df = pd.read_csv(path)

scaler = MinMaxScaler(feature_range = (0, 1))


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}


def evaluate_and_plot_confusion_matrix(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Plotting confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, title=f'Confusion Matrix: {clf.__class__.__name__}')
    plt.show()

    return accuracy, f1, precision, recall


sum_metrics = {name: {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0} for name in classifiers.keys()}

# Number of folds
n_folds = 5

# StandardScaler instance
scaler = MinMaxScaler(feature_range=(0, 1))

# Loop over folds
for fold in range(1, n_folds + 1):
    # Splitting the data into train and test sets based on fold
    train = df[df['fold'] != fold]
    test = df[df['fold'] == fold]

    X_train = train.drop(columns=['filename', 'fold', 'target', 'category'])
    y_train = train['target']

    X_test = test.drop(columns=['filename', 'fold', 'target', 'category'])
    y_test = test['target']

    # Scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate each classifier
    for name, clf in classifiers.items():
        accuracy, f1, precision, recall = evaluate_and_plot_confusion_matrix(clf, X_train_scaled, y_train,
                                                                             X_test_scaled, y_test)

        # Accumulate metrics
        sum_metrics[name]['accuracy'] += accuracy
        sum_metrics[name]['f1'] += f1
        sum_metrics[name]['precision'] += precision
        sum_metrics[name]['recall'] += recall

# Calculate and print average metrics for each classifier
for name, metrics in sum_metrics.items():
    print(f"Average metrics for {name}:")
    print(f"Accuracy: {metrics['accuracy'] / n_folds:.4f}")
    print(f"F1 Score: {metrics['f1'] / n_folds:.4f}")
    print(f"Precision: {metrics['precision'] / n_folds:.4f}")
    print(f"Recall: {metrics['recall'] / n_folds:.4f}\n")
