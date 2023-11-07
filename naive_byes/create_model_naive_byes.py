import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

def create_model_and_save(train_data, train_labels, model_filename="model.pkl", vectorizer_filename="vectorizer.pkl", vectorizer_params={}, classifier_params={}):
    """
    Create and train a Multinomial Naive Bayes model, and save it to a file.

    Parameters:
    - train_data: List of training documents.
    - train_labels: List of corresponding labels for the training documents.
    - model_filename: Filename to save the trained model.
    - vectorizer_filename: Filename to save the trained vectorizer.
    - vectorizer_params: Parameters for the TfidfVectorizer.
    - classifier_params: Parameters for the MultinomialNB classifier.

    Returns:
    - Trained MultinomialNB classifier.
    """
    # Initialize the vectorizer with specified parameters
    vectorizer = TfidfVectorizer(**vectorizer_params)

    # Transform the training data
    train_feature_vects = vectorizer.fit_transform(train_data)

    # Initialize the Multinomial Naive Bayes classifier with specified parameters
    nb_classifier = MultinomialNB(**classifier_params)

    # Fit the classifier on the training data
    nb_classifier.fit(train_feature_vects, train_labels)

    # Save the trained model to a file
    with open(model_filename, 'wb') as model_file:
        pickle.dump(nb_classifier, model_file)

    # Save the trained vectorizer to a file
    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    return nb_classifier, vectorizer

def main():
    # Load training data
    training_corpus = fetch_20newsgroups(subset='train')
    train_data, _, train_labels, _ = train_test_split(training_corpus.data, training_corpus.target, train_size=0.8, random_state=1) 

    # Create and train the model
    create_model_and_save(train_data, train_labels)

if __name__ == "__main__":
    main()
