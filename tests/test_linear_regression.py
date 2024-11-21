from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from .utils import Dataloader, MSELoss, Optimizer
from tqdm import tqdm
import pandas as pd
import argparse

def main(epochs: int, lr: float, n_splits: int):
    
    diabetes = load_diabetes()
    features, targets = diabetes.data, diabetes.target
    _, num_features = features.shape

    criterion = MSELoss()
    
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_index, test_index) in enumerate(kf.split(features)):

        # ========== ========== ========== #
        # model initialization
        # ========== ========== ========== #
        model = LinearRegression(num_features)
        optimizer = Optimizer(model, learning_rate=lr)
        # sklearn model
        sk_model = SKLinearRegression()

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
                pred = model(_features)
                loss = criterion(_features, pred, _targets[:, None])
                dw, db = criterion.backward()
                optimizer.step(dw, db)

        pred = model(val_features)
        our_grad_descent = \
            round(mean_squared_error(val_targets, pred), 2)
        
        # ========== ========== ========== #
        # training: normal equation
        # ========== ========== ========== #

        model.fit(train_features, train_targets)
        sk_model.fit(train_features, train_targets)

        # ========== ========== ========== #
        # validation
        # ========== ========== ========== #
        pred = model(val_features)
        our_normal_eq = \
            round(mean_squared_error(val_targets, pred), 2)
        
        sk_pred = sk_model.predict(val_features)
        sk_normal_eq = \
            round(mean_squared_error(val_targets, sk_pred), 2)
        
        results.append({
            'Fold': fold,
            'Our_GradientDescent': our_grad_descent,
            'Our_NormalEq': our_normal_eq,
            'Sklearn_NormalEq': sk_normal_eq
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('test_linear_regression_results.csv', index=False)

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