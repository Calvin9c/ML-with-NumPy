from random_forest import DecisionTree, RandomForest
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

def main(
    model_name: str,
    n_splits: int
):
    
    iris = load_iris()
    features, targets = iris.data, iris.target
    
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_index, test_index) in enumerate(kf.split(features)):

        # ========== ========== ========== #
        # model initialization
        # ========== ========== ========== #
        model_name = model_name.lower()
        if model_name == 'decision_tree':
            model = DecisionTree(
                        criterion = 'gini',
                        max_features = 'sqrt',
                        max_depth = None,
                        min_samples_split = 2,
                        min_impurity_decrease = 0.0
                    )
            sk_model = SKDecisionTree()
        elif model_name == 'random_forest':
            model = RandomForest(
                        num_estimators = 100, # Number of trees in the forest.
                        bootstrap = True, # Whether to use bootstrap sampling.
                        random_seed = None,
                        criterion = 'gini', # Impurity criterion for the trees ('gini' or 'entropy').
                        max_features = 'sqrt', # Maximum number of features to consider at each split.
                        max_depth = None, # Maximum depth of each tree.
                        min_samples_split = 2,
                        min_impurity_decrease = 0.0
                    )
            sk_model = SKRandomForest()
        else:
            raise NotImplementedError
        
        train_features, train_targets = features[train_index], targets[train_index]
        val_features, val_targets = features[test_index], targets[test_index]    
        
        model(train_features, train_targets)
        sk_model.fit(train_features, train_targets)

        acc = round(accuracy_score(val_targets, model(val_features)), 4)
        sk_acc = round(accuracy_score(val_targets, sk_model.predict(val_features)), 4)

        results.append({
            'Fold': fold,
            'Our': acc,
            'Sklearn': sk_acc
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('test_random_forest_results.csv', index=False) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--model_name', type=str, default='random_forest', choices=['decision_tree', 'random_forest'], help='')
    parser.add_argument('--n_splits', type=int, default=4, help='')
    
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        n_splits=args.n_splits
    )