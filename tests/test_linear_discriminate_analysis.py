from linear_discriminate_analysis import LDA, SKLDA
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

def main(
    num_components: int,
    num_neighbors: int,
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
        model = LDA(num_components=num_components, num_neighbors=num_neighbors)
        sk_model = SKLDA(num_components=num_components, num_neighbors=num_neighbors)
        
        # ========== ========== ========== #
        # set train & val data
        # ========== ========== ========== #
        train_features, train_targets = features[train_index], targets[train_index]
        val_features, val_targets = features[test_index], targets[test_index]      
        
        # ========== ========== ========== #
        # Training
        # ========== ========== ========== #
        model.fit(train_features, train_targets)
        sk_model.fit(train_features, train_targets)

        # ========== ========== ========== #
        # Validation
        # ========== ========== ========== #
        
        # predict by nearest mean
        pred_nmean = model.predict_nearest_mean(val_features)
        nmean_acc = round(accuracy_score(val_targets, pred_nmean), 4)
        
        pred_knn = model.predict_knn(val_features, train_features, train_targets)
        knn_acc = round(accuracy_score(val_targets, pred_knn), 4)
        
        sk_pred_knn = sk_model.predict_knn(val_features)
        sk_knn_acc = round(accuracy_score(val_targets, sk_pred_knn), 4)

        results.append({
            'Fold': fold,
            'Our_nmean': nmean_acc,
            'Our_kNN': knn_acc,
            'Sklearn_kNN': sk_knn_acc
        })

        model.plot(val_features, val_targets)
        sk_model.plot(val_features, val_targets)
        
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('test_lda_results.csv', index=False) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--num_components', type=int, default=2, choices=[1, 2], help='')
    parser.add_argument('--num_neighbors', type=int, default=3, help='')
    parser.add_argument('--n_splits', type=int, default=4, help='')
    
    args = parser.parse_args()    
    
    main(
        num_components=args.num_components,
        num_neighbors=args.num_neighbors,
        n_splits=args.n_splits
    )