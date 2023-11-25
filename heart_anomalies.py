import csv


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
