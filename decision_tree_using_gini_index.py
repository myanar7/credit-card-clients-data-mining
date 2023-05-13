from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def decision_tree_classification_gini(data):
    # Prepare the dataset
    X = data.drop('default payment next month', axis=1)
    y = data['default payment next month']

    # Split data into training and test sets using holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the decision tree classifier with Gini index criterion
    clf = DecisionTreeClassifier(criterion="gini", random_state=42)

    # Train the model using cross-validation on the training set
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=kf)

    # Train the model on the full training set
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    return y_train, y_train_pred, y_test, y_pred

def evaluate_model(y_train, y_train_pred, y_test, y_pred):
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_pred)

    results = {
        'train': {
            'confusion_matrix': conf_matrix_train,
            'accuracy': acc_train,
            'recall': recall_train,
            'precision': precision_train,
            'f1_score': f1_train
        },
        'test': {
            'confusion_matrix': conf_matrix_test,
            'accuracy': acc_test,
            'recall': recall_test,
            'precision': precision_test,
            'f1_score': f1_test
        }
    }

    return results
