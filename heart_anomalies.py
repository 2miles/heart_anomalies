import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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
    data = read_csv("heart-anomalies.csv")

    # Split the data into an array of features and labels
    features = [row[1:] for row in data]
    labels = [row[0] for row in data]

    # Split the data into training and testing sets
    # test_size: The proportion of the dataset to include in the test split.
    # 0.2 means 20% of the data will be used for training; 80$ for testing
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=69
    )
    # Create a decision tree classifier
    clf = DecisionTreeClassifier(random_state=69)

    # Train the classifier on the training set
    # Take the training data and labels as args and adjusts the model's parameters to learn the patterns in the data.
    # After this step, the classifier is ready to make predictions on new, unseen data.
    clf.fit(features_train, labels_train)

    # Make predictions on the test set
    labels_prediction = clf.predict(features_test)

    # Evaluate the accuracy of the classifier
    # The accuracy is the ratio of correctly predicted instances to the total number of instances in the test set.
    accuracy = accuracy_score(labels_test, labels_prediction)
    print(f"Accuracy: {accuracy * 100:.2f}%")
