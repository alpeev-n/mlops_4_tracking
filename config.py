config = {
    "random_state": 42,
    "data": {
        "test_size": 0.2,
    },
    "logistic_regression": {
        "max_iter": 200,
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
    },
    "decision_tree": {
        "max_depth": 10,
        "criterion": "gini",
        "min_samples_leaf": 1,
    }
}
