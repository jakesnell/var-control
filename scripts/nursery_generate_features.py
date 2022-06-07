import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle

nursery_data_path = 'data/nursery/'
nursery_csv_path = nursery_data_path + 'nursery.csv'
all_features = ["parents", "has_nurs", "form", "children",
                "housing", "finance", "social", "health",
                "class"]
label = "class"
protected_feature = "finance"

label_dict = {
    'parents': ['usual', 'pretentious', 'great_pret'],
    'has_nurs': ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
    'form': ['complete', 'completed', 'incomplete', 'foster'],
    'children': ['1', '2', '3', 'more'],
    'housing': ['convenient', 'less_conv', 'critical'],
    'finance': ['convenient', 'inconv'],
    'social': ['nonprob', 'slightly_prob', 'problematic'],
    'health': ['recommended', 'priority', 'not_recom'],
    'class': ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
}


def main():

    d = pd.read_csv(nursery_csv_path,
                    names=all_features)

    from collections import Counter
    print(Counter(d['finance'].tolist()))

    def f(x):
        for feature in all_features:
            labels = label_dict[feature]
            x[feature] = labels.index(x[feature])

        return x

    d = d.apply(f, axis=1)
    for column in all_features:
        d[column] = d[column].astype("category")

    X = d.loc[:, d.columns != label]
    y = d.loc[:, d.columns == label]

    test_proportion = 0.33
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_proportion,
        random_state=42)

    class_list = y_train['class'].tolist()

    while len(set(class_list)) < len(label_dict['class']):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_proportion
        )

        class_list = y_train['class'].tolist()

    X_train_status = X_train.loc[:, X_train.columns == protected_feature][protected_feature].tolist()
    X_test_status = X_test.loc[:, X_train.columns == protected_feature][protected_feature].tolist()

    X_train_non_status = X_train.loc[:, X_train.columns != protected_feature]
    X_test_non_status = X_test.loc[:, X_train.columns != protected_feature]

    pipeline = Pipeline(
        [
            ('scaling', StandardScaler()),
        ]
    )

    X_train_non_status = pipeline.fit_transform(X_train_non_status)
    X_test_non_status = pipeline.transform(X_test_non_status)

    print(Counter(
            X_test_status
        )
    )

    y_train = np.array(y_train['class'])
    y_test = np.array(y_test['class'].tolist())

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_non_status, y_train)
    logits = clf.decision_function(X_test_non_status)
    assert logits.shape[1] == len(label_dict['class'])
    score = clf.score(X_test_non_status, y_test)

    print("logits shape:", logits.shape)
    print("labels shape:", y_test.shape)

    nursery_data = {
        'z': logits,
        'y': y_test,
        'status': np.array(X_test_status),
        'label_dict': label_dict
    }
    print(score)

    # with open(nursery_data_path + "data.pkl", 'wb') as handle:
    #     pickle.dump(nursery_data, handle)


if __name__ == "__main__":
    main()
