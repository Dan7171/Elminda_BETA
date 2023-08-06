import numpy as np

scores_model = []
scores_dummy = []

for n in range(500):
    random_indices = np.random.choice(range(len(X_test)), size=len(X_test), replace=True)

    X_test_new = X_test[random_indices]

    y_test_new = y_test[random_indices]

    scores_model.append(precision_score(y_test_new, model.predict(X_test_new)))

    scores_dummy.append(precision_score(y_test_new, dummy.predict(X_test_new)))

    np.quantile(scores_model,[0.025,0.975]),np.quantile(scores_dummy,[0.025,0.975])

