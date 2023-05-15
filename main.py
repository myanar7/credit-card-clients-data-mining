import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data
from decision_tree_using_gain_ratio import decision_tree_classification_gain, evaluate_model_gain_ratio
from decision_tree_using_gini_index import decision_tree_classification_gini, evaluate_model_gini_index
from ann_using_1_hidden_layer import ann_1_hidden_layer_classification, evaluate_ann_1_hidden_layer
from ann_using_2_hidden_layers import ann_2_hidden_layers_classification, evaluate_ann_2_hidden_layers
from naive_Bayes import evaluate_model_naive_bayes, naive_bayes_classification

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
    y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain = decision_tree_classification_gain(data_clipped)

    # Evaluate model
    results_gain = evaluate_model_gain_ratio(y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)
    # Calculate error rates
    error_rate_train = 1 - results_gain['train']['accuracy']
    error_rate_test = 1 - results_gain['test']['accuracy']

    print("Decision Tree using Gain Ratio")
    print("------------------------------")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_gain['train']['confusion_matrix'])
    print("Accuracy:", results_gain['train']['accuracy'])
    print("Recall:", results_gain['train']['recall'])
    print("Precision:", results_gain['train']['precision'])
    print("F1 Score:", results_gain['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_gain['test']['confusion_matrix'])
    print("Accuracy:", results_gain['test']['accuracy'])
    print("Recall:", results_gain['test']['recall'])
    print("Precision:", results_gain['test']['precision'])
    print("F1 Score:", results_gain['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train)
    print("Error Rate (Test):", error_rate_test)

    # Decision tree using Gini index classification
    y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini = decision_tree_classification_gini(data_clipped)

    # Evaluate model
    results_gini = evaluate_model_gini_index(y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini)
    # Calculate error rates
    error_rate_train_gini = 1 - results_gini['train']['accuracy']
    error_rate_test_gini = 1 - results_gini['test']['accuracy']

    print("\nDecision Tree using Gini Index")
    print("--------------------------------")
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

    # ANN with 1 hidden layer classification
    y_train_ann_1, y_train_pred_ann_1, y_test_ann_1, y_pred_ann_1 = ann_1_hidden_layer_classification(data_clipped)

    # Evaluate model
    results_ann_1 = evaluate_ann_1_hidden_layer(y_train_ann_1, y_train_pred_ann_1, y_test_ann_1, y_pred_ann_1)

    print("\nANN with 1 Hidden Layer")
    print("-------------------------")
    print("Training set results:")
    print("Accuracy:", results_ann_1['train']['accuracy'])
    print("Recall:", results_ann_1['train']['recall'])
    print("Precision:", results_ann_1['train']['precision'])
    print("F1 Score:", results_ann_1['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_ann_1['test']['accuracy'])
    print("Recall:", results_ann_1['test']['recall'])
    print("Precision:", results_ann_1['test']['precision'])
    print("F1 Score:", results_ann_1['test']['f1_score'])

    print("\nError Rate (Training):", 1 - results_ann_1['train']['accuracy'])
    print("Error Rate (Test):", 1 - results_ann_1['test']['accuracy'])

    # ANN with 2 hidden layer classification
    y_train_ann_2, y_train_pred_ann_2, y_test_ann_2, y_pred_ann_2 = ann_2_hidden_layers_classification(data_clipped)

    # Evaluate model
    results_ann_2 = evaluate_ann_2_hidden_layers(y_train_ann_2, y_train_pred_ann_2, y_test_ann_2, y_pred_ann_2)

    print("\nANN with 2 Hidden Layer")
    print("-------------------------")
    print("Training set results:")
    print("Accuracy:", results_ann_2['train']['accuracy'])
    print("Recall:", results_ann_2['train']['recall'])
    print("Precision:", results_ann_2['train']['precision'])
    print("F1 Score:", results_ann_2['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_ann_2['test']['accuracy'])
    print("Recall:", results_ann_2['test']['recall'])
    print("Precision:", results_ann_2['test']['precision'])
    print("F1 Score:", results_ann_2['test']['f1_score'])

    print("\nError Rate (Training):", 1 - results_ann_2['train']['accuracy'])
    print("Error Rate (Test):", 1 - results_ann_2['test']['accuracy'])
# Naive Bayes classification
    y_train_naive, y_train_pred_naive, y_test_naive, y_pred_naive = naive_bayes_classification(data_clipped)

# Evaluate model
    results_naive = evaluate_model_naive_bayes(y_train_naive, y_train_pred_naive, y_test_naive, y_pred_naive)

    print("\nNaive Bayes Classification")
    print("-------------------------")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_naive['train']['confusion_matrix'])
    print("Accuracy:", results_naive['train']['accuracy'])
    print("Recall:", results_naive['train']['recall'])
    print("Precision:", results_naive['train']['precision'])
    print("F1 Score:", results_naive['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_naive['test']['confusion_matrix'])
    print("Accuracy:", results_naive['test']['accuracy'])
    print("Recall:", results_naive['test']['recall'])
    print("Precision:", results_naive['test']['precision'])
    print("F1 Score:", results_naive['test']['f1_score'])

    print("\nError Rate (Training):", 1 - results_naive['train']['accuracy'])
    print("Error Rate (Test):", 1 - results_naive['test']['accuracy'])    
if __name__ == "__main__":
    main()