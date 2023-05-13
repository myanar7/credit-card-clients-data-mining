import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data
from decision_tree_using_gain_ratio import decision_tree_classification, evaluate_model
from decision_tree_using_gini_index import decision_tree_classification_gini

def main():
    # Load and preprocess data
    data = load_data("data.xls")
    data_clipped = preprocess_data(data)

    # Create figures folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Plot correlation matrices
    # Saving the correlation matrix of the original data
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Using Original Data")
    plt.savefig(os.path.join("figures", "correlation_matrix_original.png"), dpi=300)
    plt.close()

    # Saving the correlation matrix of data free of outliers
    corr_clipped = data_clipped.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr_clipped, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Using Clipped Data")
    plt.savefig(os.path.join("figures", "correlation_matrix_clipped.png"), dpi=300)
    plt.close()

    # Decision tree using gain ratio classification
    y_train, y_train_pred, y_test, y_pred = decision_tree_classification(data_clipped)

    # Evaluate model
    results = evaluate_model(y_train, y_train_pred, y_test, y_pred)
    # Calculate error rates
    error_rate_train = 1 - results['train']['accuracy']
    error_rate_test = 1 - results['test']['accuracy']

    print("Decision Tree using Gain Ratio")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results['train']['confusion_matrix'])
    print("Accuracy:", results['train']['accuracy'])
    print("Recall:", results['train']['recall'])
    print("Precision:", results['train']['precision'])
    print("F1 Score:", results['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results['test']['confusion_matrix'])
    print("Accuracy:", results['test']['accuracy'])
    print("Recall:", results['test']['recall'])
    print("Precision:", results['test']['precision'])
    print("F1 Score:", results['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train)
    print("Error Rate (Test):", error_rate_test)

    # Decision tree using Gini index classification
    y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini = decision_tree_classification_gini(data_clipped)

    # Evaluate model
    results_gini = evaluate_model(y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini)
    # Calculate error rates
    error_rate_train_gini = 1 - results_gini['train']['accuracy']
    error_rate_test_gini = 1 - results_gini['test']['accuracy']

    print("\nDecision Tree using Gini Index")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_gini['train']['confusion_matrix'])
    print("Accuracy:", results_gini['train']['accuracy'])
    print("Recall:", results_gini['train']['recall'])
    print("Precision:", results_gini['train']['precision'])
    print("F1 Score:", results_gini['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_gini['test']['confusion_matrix'])
    print("Accuracy:", results_gini['test']['accuracy'])
    print("Recall:", results_gini['test']['recall'])
    print("Precision:", results_gini['test']['precision'])
    print("F1 Score:", results_gini['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_gini)
    print("Error Rate (Test):", error_rate_test_gini)

if __name__ == "__main__":
    main()