import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the classes from utils.py
from utils import ClassificationPipeline, StandardNNClassifier

def main():
    # Load the Iris dataset.
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------
    # Train and evaluate all three SVM+FFNN pipelines.
    # ---------------------------
    accuracies = {}
    
    for svm_target in [0, 1, 2]:
        pipeline = ClassificationPipeline(svm_target=svm_target)
        pipeline.fit(X_train, y_train)
        pipeline_preds = pipeline.predict(X_test)
        pipeline_accuracy = np.mean(pipeline_preds == y_test)
        accuracies[f"Pipeline (SVM target {svm_target})"] = pipeline_accuracy * 100
        print(f"SVM+FFNN Pipeline (Target {svm_target}) Test Accuracy: {pipeline_accuracy * 100:.2f}%")

    # ---------------------------
    # Train and evaluate the standard NN classifier.
    # ---------------------------
    standard_nn = StandardNNClassifier()
    standard_nn.fit(X_train, y_train)
    standard_nn_preds = standard_nn.predict(X_test)
    standard_nn_accuracy = np.mean(standard_nn_preds == y_test)
    accuracies["Standard NN"] = standard_nn_accuracy * 100
    print("Standard NN Test Accuracy: {:.2f}%".format(standard_nn_accuracy * 100))


if __name__ == "__main__":
    main()