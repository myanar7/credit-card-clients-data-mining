import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data
from decision_tree_using_gain_ratio import decision_tree_classification, evaluate_model

def main():
    # Load and preprocess data
    data = load_data("data.xls")
    data_clipped = preprocess_data(data)

    # Create figures folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Plot correlation matrices
    # Orijinal verilerin korelasyon matrisini kaydetme
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Using Original Data")
    plt.savefig(os.path.join("figures", "correlation_matrix_original.png"), dpi=300)
    plt.close()

    # Aykırı değerlerden arındırılmış verilerin korelasyon matrisini kaydetme
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

if __name__ == "__main__":
    main()
