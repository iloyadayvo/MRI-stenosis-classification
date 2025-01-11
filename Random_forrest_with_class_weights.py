import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, roc_curve, auc, precision_recall_curve, mean_squared_error)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_data(data_dir, categories):
    X = []
    y = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_file in os.listdir(category_path):
            img = imread(os.path.join(category_path, img_file))
            img_resized = resize(img, (128, 128))
            X.append(img_resized.flatten())
            y.append(categories.index(category))
    return np.array(X), np.array(y)


def calculate_class_weights(y):
    # Compute class weights inversely proportional to class frequencies
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(zip(np.unique(y), class_weights))


def plot_roc_curves(y_test_bin, y_pred_proba, categories):
    plt.figure(figsize=(10, 8))

    for i in range(len(categories)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{categories[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.savefig("Weighted_Random_Forest_ROC_curves.png")
    plt.show()


def plot_feature_importance(rf_model, pca):
    feature_importance = rf_model.feature_importances_
    pca_components = np.arange(len(feature_importance))

    plt.figure(figsize=(10, 6))
    plt.bar(pca_components, feature_importance)
    plt.xlabel('Principal Components')
    plt.ylabel('Feature Importance')
    plt.title('Weighted Random Forest Feature Importance')
    plt.savefig("Weighted_Random_Forest_Feature_Importance.png")
    plt.show()





def plot_confusion_matrix(conf_matrix, categories):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Weighted Random Forest)")
    plt.savefig("Weighted_Random_Forest_confusion_matrix.png")
    plt.show()





# [Previous functions remain the same until plot_confusion_matrix]

def calculate_rmse_per_class(y_true_bin, y_pred_proba):
    """Calculate RMSE for each class."""
    rmse_per_class = []
    for i in range(y_true_bin.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true_bin[:, i], y_pred_proba[:, i]))
        rmse_per_class.append(rmse)
    return rmse_per_class


def plot_rmse(rmse_per_class, categories):
    """Plot RMSE for each class."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, rmse_per_class)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.xlabel('Classes')
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Weighted_Random_Forest_RMSE.png")
    plt.show()


def main():
    # Setup
    data_dir = 'C:/Users/ASUS/Desktop/Mrs Awoniran MRI classification/classification spinal canal stenosis'
    categories = ['Moderate', 'Normal', 'Severe']

    # Load data
    X, y = load_data(data_dir, categories)

    # Calculate class weights
    class_weights = calculate_class_weights(y)
    print("\nClass Weights:")
    for category, weight in zip(categories, class_weights.values()):
        print(f"{category}: {weight:.2f}")

    # PCA reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Split data
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Weighted Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weights,
        random_state=42
    )
    rf_model.fit(X_train_pca, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_pca)
    y_pred_proba = rf_model.predict_proba(X_test_pca)

    # Prepare for metrics
    label_binarizer = LabelBinarizer()
    y_test_bin = label_binarizer.fit_transform(y_test)
    if len(categories) == 2:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    # Calculate metrics including RMSE
    cross_entropy_per_class = []
    for i in range(len(categories)):
        ce = log_loss(y_test_bin[:, i], y_pred_proba[:, i])
        cross_entropy_per_class.append(ce)

    cross_entropy = log_loss(y_test_bin, y_pred_proba)
    rmse_per_class = calculate_rmse_per_class(y_test_bin, y_pred_proba)
    overall_rmse = np.sqrt(mean_squared_error(y_test_bin, y_pred_proba))

    # Print results
    print("\nWeighted Random Forest Results:")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"\nOverall RMSE: {overall_rmse:.3f}")
    print("\nRMSE per class:")
    for category, rmse in zip(categories, rmse_per_class):
        print(f"{category}: {rmse:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Generate visualizations
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, categories)
    plot_roc_curves(y_test_bin, y_pred_proba, categories)
    plot_feature_importance(rf_model, pca)
    plot_rmse(rmse_per_class, categories)

    # Save the model

    with open('C:/Users/ASUS/Desktop/Mrs Awoniran MRI classification/classification spinal canal stenosis/random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)


if __name__ == "__main__":
    main()