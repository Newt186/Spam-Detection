# Spam Detection

This project is a Machine Learning based spam detection system. It classifies text messages as either Spam or Not Spam.

## Dataset

The project uses a labeled SMS spam dataset containing two types of messages:

* Ham - Normal message
* Spam - Unwanted or promotional message

## Model Used

The project uses:

* TF-IDF Vectorizer
* Naive Bayes Classifier

TF-IDF is used to convert text messages into numerical features that can be used by the Machine Learning model.

## Steps

1. Load the spam dataset
2. Clean the text data
3. Convert labels into numerical values
4. Split the dataset into training and testing data
5. Convert text into vectors using TF-IDF
6. Train the Naive Bayes model
7. Evaluate the model
8. Take a new message as input
9. Predict whether the message is Spam or Not Spam

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

## Technologies Used

* Python
* Pandas
* Scikit-learn

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Make sure `spam.csv` is present in the project folder.

Then run:

```bash
python spam_detection.py
```

## Example

Input:

```text
Congratulations! You have won a free prize. Click now!
```

Output:

```text
Spam
```

Another example:

```text
Hey, are you coming to college tomorrow?
```

Output:

```text
Not Spam
```

## Project Purpose

The main purpose of this project is to understand basic Natural Language Processing and Machine Learning classification using text data.

