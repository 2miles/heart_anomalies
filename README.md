# Heart Anomalies

### Miles Whitaker

Homework 3: Machine Learning Question 1.
CS-441 AI, Nov 2023, Bart Massey

The background for this exercise is is described here: https://github.com/pdx-cs-ai/heart-anomaly-hw

Given a data set with 267 instances. Each instance is a binary label representing presence or absence of a heart anomaly, followed by a list of binary features.

We are to construct a machine learner that can achieve at least 70% accuracy in classifying instances as anomalous vs normal

### What I did

I decided to use the scikit-learn ML framewok to build this decision tree classifier. Most of my time was spent reading the documentation for the necessary scikit modules, then visiting https://datagy.io/sklearn-decision-tree-classifier/ to get an idea of how one could be implemented.

### What is still to be done

- I would like to code this up without using a framework.

## Running

You are going to need to have `scikit-learn` installed

To run the program with all the default parameters:

```
python3 heart_anomalies.py
```

### Arguments:

#### `--custom_split [float]`

The percentage of data to be included in test set. Decimal between 0 and 1. If a custom_split if given then no cross-validation will be done. The default value is `None`

EX: `0.2` will use 20% of the data for learning and 80% for testing.

#### `--seed [int]`

Seed used for the random number generator for the test/train splitting algorithm and for the decision tree algorithm. The default value is arbitrarily set to `50`.

#### `--cross_val [int]`

The number of subsets into which the data is divided for cross validation. The default value is `5`
