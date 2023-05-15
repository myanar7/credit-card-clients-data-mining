import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import load_data, preprocess_data
from decision_tree_using_gain_ratio import decision_tree_classification_gain, evaluate_model_gain_ratio
from decision_tree_using_gini_index import decision_tree_classification_gini, evaluate_model_gini_index
from ann_using_1_hidden_layer import ann_1_hidden_layer_classification, evaluate_ann_1_hidden_layer
from ann_using_2_hidden_layers import ann_2_hidden_layers_classification, evaluate_ann_2_hidden_layers
from svm import svm_classification, evaluate_model_svm
from naive_Bayes import naive_bayes_classification, evaluate_model_naive_bayes

def main():
    # Load and preprocess data
    data = load_data("data.xls")
    data_clipped = preprocess_data(data)

    # Create the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to the clipped data and transform it
    data_normalized = scaler.fit_transform(data_clipped)

    # Convert the numpy array back to a DataFrame
    data_normalized = pd.DataFrame(data_normalized, columns=data_clipped.columns)


    # Create figures folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Visualize results
    # List of models
    models = ['Gain Ratio', 'Gini Index', 'Naive Bayes', 'ANN 1 Layer', 'ANN 2 Layers', 'SVM']

    # Plot correlation matrices
    # Saving the correlation matrix of the original data
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Using Original Data")
    plt.savefig(os.path.join("figures", "correlation_matrix_original.png"), dpi=300)
    plt.close()

    # Saving the correlation matrix of data free of outliers
    corr_clipped = data_normalized.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr_clipped, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Using Clipped Data")
    plt.savefig(os.path.join("figures", "correlation_matrix_clipped.png"), dpi=300)
    plt.close()

    # Decision tree using gain ratio classification
    y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain = decision_tree_classification_gain(data_normalized)

    # Evaluate model
    results_gain = evaluate_model_gain_ratio(y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)
    # Calculate error rates
    error_rate_train_gain = 1 - results_gain['train']['accuracy']
    error_rate_test_gain = 1 - results_gain['test']['accuracy']

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

    print("\nError Rate (Training):", error_rate_train_gain)
    print("Error Rate (Test):", error_rate_test_gain)

    # Decision tree using Gini index classification
    y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini = decision_tree_classification_gini(data_normalized)

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

    # Naive Bayes classification
    y_train_naive, y_train_pred_naive, y_test_naive, y_pred_naive = naive_bayes_classification(data_normalized)

    # Evaluate model
    results_naive = evaluate_model_naive_bayes(y_train_naive, y_train_pred_naive, y_test_naive, y_pred_naive)
    # Calculate error rates
    error_rate_train_naive = 1 - results_naive['train']['accuracy']
    error_rate_test_naive = 1 - results_naive['test']['accuracy']

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

    print("\nError Rate (Training):", error_rate_train_naive)
    print("Error Rate (Test):", 1 - error_rate_test_naive)  

    # ANN with 1 hidden layer classification
    y_train_ann_1, y_train_pred_ann_1, y_test_ann_1, y_pred_ann_1 = ann_1_hidden_layer_classification(data_normalized)

    # Evaluate model
    results_ann_1 = evaluate_ann_1_hidden_layer(y_train_ann_1, y_train_pred_ann_1, y_test_ann_1, y_pred_ann_1)
    # Calculate error rates
    error_rate_train_ann1 = 1 - results_ann_1['train']['accuracy']
    error_rate_test_ann1 = 1 - results_ann_1['test']['accuracy']

    print("\nANN with 1 Hidden Layer")
    print("-------------------------")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_ann_1['train']['confusion_matrix'])
    print("Accuracy:", results_ann_1['train']['accuracy'])
    print("Recall:", results_ann_1['train']['recall'])
    print("Precision:", results_ann_1['train']['precision'])
    print("F1 Score:", results_ann_1['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_ann_1['test']['confusion_matrix'])
    print("Accuracy:", results_ann_1['test']['accuracy'])
    print("Recall:", results_ann_1['test']['recall'])
    print("Precision:", results_ann_1['test']['precision'])
    print("F1 Score:", results_ann_1['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_ann1)
    print("Error Rate (Test):", error_rate_test_ann1)

    # ANN with 2 hidden layer classification
    y_train_ann_2, y_train_pred_ann_2, y_test_ann_2, y_pred_ann_2 = ann_2_hidden_layers_classification(data_normalized)

    # Evaluate model
    results_ann_2 = evaluate_ann_2_hidden_layers(y_train_ann_2, y_train_pred_ann_2, y_test_ann_2, y_pred_ann_2)
    # Calculate error rates
    error_rate_train_ann2 = 1 - results_ann_2['train']['accuracy']
    error_rate_test_ann2 = 1 - results_ann_2['test']['accuracy']

    print("\nANN with 2 Hidden Layer")
    print("-------------------------")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_ann_2['train']['confusion_matrix'])
    print("Accuracy:", results_ann_2['train']['accuracy'])
    print("Recall:", results_ann_2['train']['recall'])
    print("Precision:", results_ann_2['train']['precision'])
    print("F1 Score:", results_ann_2['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_ann_2['test']['confusion_matrix'])
    print("Accuracy:", results_ann_2['test']['accuracy'])
    print("Recall:", results_ann_2['test']['recall'])
    print("Precision:", results_ann_2['test']['precision'])
    print("F1 Score:", results_ann_2['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_ann2)
    print("Error Rate (Test):", error_rate_test_ann2)

    # Support Vector Machines layer classification
    y_train_svm, y_train_pred_svm, y_test_svm, y_pred_svm = svm_classification(data_normalized)

    # Evaluate model
    results_svm = evaluate_model_svm(y_train_svm, y_train_pred_svm, y_test_svm, y_pred_svm)
    # Calculate error rates
    error_rate_train_svm = 1 - results_svm['train']['accuracy']
    error_rate_test_svm = 1 - results_svm['test']['accuracy']

    print("\nSVM Classification")
    print("-------------------------")
    print("Training set results:")
    print("Confusion Matrix:")
    print(results_svm['train']['confusion_matrix'])
    print("Accuracy:", results_svm['train']['accuracy'])
    print("Recall:", results_svm['train']['recall'])
    print("Precision:", results_svm['train']['precision'])
    print("F1 Score:", results_svm['train']['f1_score'])

    print("\nTest set results:")
    print("Confusion Matrix:")
    print(results_svm['test']['confusion_matrix'])
    print("Accuracy:", results_svm['test']['accuracy'])
    print("Recall:", results_svm['test']['recall'])
    print("Precision:", results_svm['test']['precision'])
    print("F1 Score:", results_svm['test']['f1_score'])
    
    print("\nError Rate (Training):", error_rate_train_svm)
    print("Error Rate (Test):", error_rate_test_svm)

    # Visualize results
    # Visualize results of accuracy
    # Gather metrics into dictionaries
    train_accuracy = {
        'Gain Ratio': results_gain['train']['accuracy'],
        'Gini Index': results_gini['train']['accuracy'],
        'Naive Bayes': results_naive['train']['accuracy'],
        'ANN 1 Layer': results_ann_1['train']['accuracy'],
        'ANN 2 Layers': results_ann_2['train']['accuracy'],
        'SVM': results_svm['train']['accuracy']
    }

    test_accuracy = {
        'Gain Ratio': results_gain['test']['accuracy'],
        'Gini Index': results_gini['test']['accuracy'],
        'Naive Bayes': results_naive['test']['accuracy'],
        'ANN 1 Layer': results_ann_1['test']['accuracy'],
        'ANN 2 Layers': results_ann_2['test']['accuracy'],
        'SVM': results_svm['test']['accuracy']
    }

    # Create bar plots
    labels = list(train_accuracy.keys())
    train_values = list(train_accuracy.values())
    test_values = list(test_accuracy.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("figures", "accuracy_comparison.png"), dpi=300)
    plt.close()


    # Visualize results of recall
    # Gather metrics into dictionaries
    train_recall = {
        'Gain Ratio': results_gain['train']['recall'],
        'Gini Index': results_gini['train']['recall'],
        'Naive Bayes': results_naive['train']['recall'],
        'ANN 1 Layer': results_ann_1['train']['recall'],
        'ANN 2 Layers': results_ann_2['train']['recall'],
        'SVM': results_svm['train']['recall']
    }

    test_recall = {
        'Gain Ratio': results_gain['test']['recall'],
        'Gini Index': results_gini['test']['recall'],
        'Naive Bayes': results_naive['test']['recall'],
        'ANN 1 Layer': results_ann_1['test']['recall'],
        'ANN 2 Layers': results_ann_2['test']['recall'],
        'SVM': results_svm['test']['recall']
    }

    # Create bar plots
    labels = list(train_recall.keys())
    train_values = list(train_recall.values())
    test_values = list(test_recall.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('Recall')
    ax.set_title('Recall by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("figures", "recall_comparison.png"), dpi=300)
    plt.close()

    # Visualize results of precision
    # Gather metrics into dictionaries
    train_precision = {
        'Gain Ratio': results_gain['train']['precision'],
        'Gini Index': results_gini['train']['precision'],
        'Naive Bayes': results_naive['train']['precision'],
        'ANN 1 Layer': results_ann_1['train']['precision'],
        'ANN 2 Layers': results_ann_2['train']['precision'],
        'SVM': results_svm['train']['precision']
    }

    test_precision = {
        'Gain Ratio': results_gain['test']['precision'],
        'Gini Index': results_gini['test']['precision'],
        'Naive Bayes': results_naive['test']['precision'],
        'ANN 1 Layer': results_ann_1['test']['precision'],
        'ANN 2 Layers': results_ann_2['test']['precision'],
        'SVM': results_svm['test']['precision']
    }

    # Create bar plots
    labels = list(train_precision.keys())
    train_values = list(train_precision.values())
    test_values = list(test_precision.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('Precision')
    ax.set_title('Precision by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("figures", "precision_comparison.png"), dpi=300)
    plt.close()

    # Visualize results of f1_score
    # Gather metrics into dictionaries
    train_f1_score = {
        'Gain Ratio': results_gain['train']['f1_score'],
        'Gini Index': results_gini['train']['f1_score'],
        'Naive Bayes': results_naive['train']['f1_score'],
        'ANN 1 Layer': results_ann_1['train']['f1_score'],
        'ANN 2 Layers': results_ann_2['train']['f1_score'],
        'SVM': results_svm['train']['f1_score']
    }

    test_f1_score = {
        'Gain Ratio': results_gain['test']['f1_score'],
        'Gini Index': results_gini['test']['f1_score'],
        'Naive Bayes': results_naive['test']['f1_score'],
        'ANN 1 Layer': results_ann_1['test']['f1_score'],
        'ANN 2 Layers': results_ann_2['test']['f1_score'],
        'SVM': results_svm['test']['f1_score']
    }

    # Create bar plots
    labels = list(train_f1_score.keys())
    train_values = list(train_f1_score.values())
    test_values = list(test_f1_score.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("figures", "f1_score_comparison.png"), dpi=300)
    plt.close()

    # Visualize results of error rates
    # Gather metrics into dictionaries
    train_error_rate = {
        'Gain Ratio': error_rate_train_gain,
        'Gini Index': error_rate_train_gini,
        'Naive Bayes': error_rate_train_naive,
        'ANN 1 Layer': error_rate_train_ann1,
        'ANN 2 Layers': error_rate_train_ann2,
        'SVM': error_rate_train_svm
    }

    test_error_rate = {
        'Gain Ratio': error_rate_test_gain,
        'Gini Index': error_rate_test_gini,
        'Naive Bayes': error_rate_test_naive,
        'ANN 1 Layer': error_rate_test_ann1,
        'ANN 2 Layers': error_rate_test_ann2,
        'SVM': error_rate_test_svm
    }

    # Create bar plots
    labels = list(train_error_rate.keys())
    train_values = list(train_error_rate.values())
    test_values = list(test_error_rate.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("figures", "error_rate_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()