import logging
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import pickle

logging.basicConfig(level=logging.INFO)

def load_model_and_vectorizer(model_filename="model.pkl", vectorizer_filename="vectorizer.pkl"):
    """
    Load a pre-trained Multinomial Naive Bayes model and vectorizer from files.

    Parameters:
    - model_filename: Filename of the trained model.
    - vectorizer_filename: Filename of the trained vectorizer.

    Returns:
    - Trained MultinomialNB classifier and vectorizer.
    """
    with open(model_filename, 'rb') as model_file:
        nb_classifier = pickle.load(model_file)

    with open(vectorizer_filename, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return nb_classifier, vectorizer

def improve_model(existing_model, existing_vectorizer, train_data, train_labels, val_data, val_labels, improved_model_filename="model_v2.pkl"):
    """
    Improve the existing model, save the improved model to a file, and return the improved model.

    Parameters:
    - existing_model: Trained classifier (e.g., MultinomialNB).
    - existing_vectorizer: Trained vectorizer.
    - train_data: List of training documents.
    - train_labels: List of corresponding labels for the training documents.
    - val_data: List of validation documents.
    - val_labels: List of corresponding labels for the validation documents.
    - improved_model_filename: Filename to save the improved model.

    Returns:
    - Improved MultinomialNB classifier.
    """
    try:
        # Re-vectorize the training set with the existing vectorizer
        train_feature_vects = existing_vectorizer.transform(train_data)

        # Check the number of features
        logging.info(f'Number of training features: {len(train_feature_vects[0].toarray().flatten())}')

        # Retrain the classifier on the updated training set
        existing_model.fit(train_feature_vects, train_labels)

        # Save the improved model to a file
        with open(improved_model_filename, 'wb') as improved_model_file:
            pickle.dump(existing_model, improved_model_file)

        # Transform the validation set with the updated vectorizer
        val_feature_vects = existing_vectorizer.transform(val_data)

        # Check validation F1 score with the improved features
        val_preds = existing_model.predict(val_feature_vects)
        logging.info(f'Validation F1 score with improved features: {metrics.f1_score(val_labels, val_preds, average="macro")}')

        # Display the confusion matrix
        fig, ax = plt.subplots(figsize=(15, 15))
        disp = ConfusionMatrixDisplay.from_estimator(existing_model, val_feature_vects, val_labels, normalize='true', xticks_rotation='vertical', ax=ax)

        # Display the classification report
        logging.info(metrics.classification_report(val_labels, val_preds))

        return existing_model
    except Exception as e:
        logging.error(f"Error improving model: {e}")
        raise

def main():
    # Load training data
    training_corpus = fetch_20newsgroups(subset='train')
    _, val_data, _, val_labels = train_test_split(training_corpus.data, training_corpus.target, train_size=0.8, random_state=1) 

    # Load the pre-trained model and vectorizer
    nb_classifier, existing_vectorizer = load_model_and_vectorizer()

    # Improve and save the model
    improved_classifier = improve_model(nb_classifier, existing_vectorizer, training_corpus.data, training_corpus.target, val_data, val_labels)

if __name__ == "__main__":
    main()
