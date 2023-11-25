import argparse
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

## This is the site I used to learn ho to make a decision tree classifier in sklearn
## https://datagy.io/sklearn-decision-tree-classifier/


def read_csv(csv_file):
    """
    Read from CSV file into a data array \\
    Return the data array
    """
    data = []
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


if __name__ == "__main__":
    # Process arguments.
    parser = argparse.ArgumentParser(description="Heart Anomaly Classifier.")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Percentage of data to be included in test set. Integer: [0,100]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the test/train splitting and the decision tree algorithm",
    )
    parser.add_argument(
        "--cross_val",
        type=int,
        default=5,
        help="Number of subset into which the data is divided for cross validation",
    )
    args = parser.parse_args()
    TEST_SIZE = args.test_size
    SEED = args.seed
    CROSS_VAL = args.cross_val

    data = read_csv("heart-anomalies.csv")

    # Split the data into an array of features and labels
    features = [row[1:] for row in data]
    labels = [row[0] for row in data]

    # Split the data into training and testing sets
    # test_size: The proportion of the dataset to include in the test split.
    # 0.2 means 20% of the data will be used for training; 80$ for testing
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=SEED
    )
    # Create a decision tree classifier
    clf = DecisionTreeClassifier(random_state=SEED)

    # Train the classifier on the training set
    # Take the training data and labels as args and adjusts the model's parameters to learn the patterns in the data.
    # After this step, the classifier is ready to make predictions on new, unseen data.
    clf.fit(features_train, labels_train)

    # Make predictions on the test set
    labels_prediction = clf.predict(features_test)

    # Evaluate the accuracy of the classifier
    # The accuracy is the ratio of correctly predicted instances to the total number of instances in the test set.
    ## accuracy = accuracy_score(labels_test, labels_prediction)
    ## print(f"Accuracy: {accuracy * 100:.2f}%")

    # n-fold cross-validation
    accuracy_scores = cross_val_score(clf, features, labels, cv=CROSS_VAL)
    # Print the accuracy for each fold

    for i in range(len(accuracy_scores)):
        accuracy = accuracy_scores[i]
        print(f"Subset: {i + 1},  Accuracy: {accuracy * 100:.2f}%")

    print(f"Average Accuracy: {accuracy_scores.mean() * 100:.2f}%")
