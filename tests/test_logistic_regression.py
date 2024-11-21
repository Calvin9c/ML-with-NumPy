from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from .utils import Dataloader, CrossEntropyLoss, Optimizer
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np

def main(epochs: int, lr: float, n_splits: int):
    
    iris = load_iris()
    features, targets = iris.data, iris.target
    _, num_features = features.shape
    num_classes = len(np.unique(targets))
    
    criterion = CrossEntropyLoss()
    
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_index, test_index) in enumerate(kf.split(features)):
    
        # ========== ========== ========== #
        # model initialization
        # ========== ========== ========== #
        model = LogisticRegression(num_features, num_classes)
        optimizer = Optimizer(model, learning_rate=lr)
        # sklearn model
        sk_model = SKLogisticRegression(max_iter=epochs)

        # ========== ========== ========== #
        # set train & val data
        # ========== ========== ========== #
        train_features, train_targets = features[train_index], targets[train_index]
        val_features, val_targets = features[test_index], targets[test_index]        
        train_loader = Dataloader(
            train_features, train_targets,
            batch_size=32,
            shuffle=True
        )

        # ========== ========== ========== #
        # training: gradient descent
        # ========== ========== ========== #
        for epoch in tqdm(range(epochs)):
            for _features, _targets in train_loader:
                pred = model(_features) # [BS, num_classes]
                loss = criterion(_features, pred, _targets)
                dw, db = criterion.backward()
                optimizer.step(dw, db)
        
        sk_model.fit(train_features, train_targets)
                
        # ========== ========== ========== #
        # validation
        # ========== ========== ========== #
        pred = model.predict(val_features)
        acc = round(accuracy_score(val_targets, pred), 4)
        
        sk_pred = sk_model.predict(val_features)
        sk_acc = round(accuracy_score(val_targets, sk_pred), 4)

        results.append({
            'Fold': fold,
            'Our': acc,
            'Sklearn': sk_acc
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('test_logistic_regression_results.csv', index=False) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epochs', type=int, default=100000, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--n_splits', type=int, default=4, help='')
    
    args = parser.parse_args()
    
    main(
        epochs=args.epochs,
        lr=args.lr,
        n_splits=args.n_splits
    )