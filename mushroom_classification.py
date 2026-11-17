from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def simplest_model(results):
    # this function returns the index of the simplest model that achieves a perfect recall score. (no poisonous mushrooms are misclassified as edible)
    # this approach is based on the assumption that the model with the fewest features is the simplest and quickest one.
    df = pd.DataFrame(results)
    safe_model = df[df['mean_test_score'] == 1.0]
    if safe_model.empty:
        return int(df['mean_test_score'].idxmax())
    k_values = safe_model['param_feature_selection__k']
    best_idx = k_values.idxmin()
    return int(best_idx)

def results_table(parameters, pipeline, x_train, y_train, x_test, y_test):
 # this function shows the table of results for different values of k in the feature selection step, including accuracy, precision, recall, and confusion matrix values.
 # the most important metrics for this problem are recall and false negatives, as their error can lead to serious consequences (poisoning). 
    results = []

    for k in parameters['feature_selection__k']:
        pipeline.set_params(feature_selection__k=k)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        results.append({'k': k,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall, 
                    'tn': tn, 
                    'fp': fp, 
                    'fn': fn, 
                    'tp': tp})

    results_df = pd.DataFrame(results).round(4)
    print("Results for different values of k:")
    print(results_df)

def plot_feature_importances(best_tree, columns):
    # this function plots the feature importances of the best decision tree model, showing which features were most influential in making predictions.
    importances = best_tree.feature_importances_
    importance_series = pd.Series(importances, index=columns)
    best_features = importance_series[importance_series > 0].sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    best_features.plot(kind='barh', color='teal')
    plt.title(f'Kluczowe cechy w optymalnym modelu (wybrano k={grid_search.best_params_["feature_selection__k"]})')
    plt.xlabel('Ważność (Gini Importance)')
    plt.ylabel('Cecha')
    plt.tight_layout()
    plt.show()

def plot_decision_tree(best_tree, columns):
    # this function plots the decision tree of the best model, providing a visual representation of how decisions are made based on feature values.
    plt.figure(figsize=(12, 8))
    tree.plot_tree(best_tree, feature_names=columns, class_names=['edible', 'poisonous'], filled=True, rounded=True, max_depth=3)
    plt.title('Drzewo decyzyjne dla najlepszego modelu')
    plt.show()



if __name__ == "__main__":
    # Load the mushroom dataset from the UCI Machine Learning Repository
    mushroom = fetch_ucirepo(id=73)
    # Prepare the data
    X = mushroom.data.features
    y = mushroom.data.targets
    mushroom_df = pd.DataFrame(X, columns=mushroom.data.feature_names)
    print(mushroom_df.shape)
    # Print columns with missing values
    print(mushroom_df.isna().sum()[mushroom_df.isna().sum() > 0])
      # Print the number of unique values in each column
    print(mushroom_df.nunique()[mushroom_df.nunique() == 1])
    # In this case, the missing values do not provide any useful information for the model, so we will drop the columns with missing values and the column with only one unique value.
    mushroom_df.drop(columns=["stalk-root","veil-type"], inplace=True)
    print(mushroom_df.shape)

    # Convert categorical features to numerical using one-hot encoding
    mushroom_df = pd.get_dummies(mushroom_df,drop_first=True)

    # Convert target variable to edible:negative (0) and poisonous:positive (1)
    y = y["poisonous"].map({"e": 0, "p": 1})

    x_train, x_test, y_train, y_test = train_test_split(mushroom_df, y, test_size=0.2, random_state=42)

    # Define a pipeline with feature selection and decision tree classifier
    pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', tree.DecisionTreeClassifier(random_state=42))
    ])

    # Define the parameter grid for GridSearchCV to find the best number of features (k) to select
    parameters = {'feature_selection__k': [5, 10, 15, 20, mushroom_df.shape[1]]}   

    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='recall', refit=simplest_model, n_jobs=-1)

    grid_search.fit(x_train, y_train)

    results_table(parameters, pipeline, x_train, y_train, x_test, y_test)

    y_pred = grid_search.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = recall_score(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Recall: {recall:.4f}")
    print(f"FN: {fn}")

    plot_feature_importances(grid_search.best_estimator_.named_steps['classifier'], mushroom_df.columns[grid_search.best_estimator_.named_steps['feature_selection'].get_support(indices=True)])

    plot_decision_tree(grid_search.best_estimator_.named_steps['classifier'], mushroom_df.columns[grid_search.best_estimator_.named_steps['feature_selection'].get_support(indices=True)])










