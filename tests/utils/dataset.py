from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

def load_dataset(
    dataset_name: str,
    test_size: float = 0.2, 
    random_seed: int = 0
):

    if dataset_name.lower() == 'iris':
        data = load_iris()
    elif dataset_name.lower() == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError("Invalid dataset name. Choose either 'iris' or 'diabetes'.")
    
    features, targets = data.data, data.target
    train_features, val_features, train_targets, val_targets = train_test_split(
        features, targets, test_size=test_size, random_state=random_seed
    )
    return train_features, train_targets, val_features, val_targets