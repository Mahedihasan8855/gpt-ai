import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

def load_model_and_vectorizer(model_filename="model_v2.pkl", vectorizer_filename="vectorizer.pkl"):
    """
    Load the pre-trained model and vectorizer from files.

    Parameters:
    - model_filename: Filename of the pre-trained model.
    - vectorizer_filename: Filename of the pre-trained vectorizer.

    Returns:
    - Tuple containing the loaded model and vectorizer.
    """
    with open(model_filename, 'rb') as model_file:
        loaded_nb_classifier = pickle.load(model_file)

    with open(vectorizer_filename, 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    return loaded_nb_classifier, loaded_vectorizer

def classify_text(classifier, vectorizer, doc, labels=None):
    """
    Classify a text document using the provided classifier and vectorizer.

    Parameters:
    - classifier: Trained classifier.
    - vectorizer: Trained vectorizer.
    - doc: Text document to classify.
    - labels: Optional list of labels for interpretation.

    Returns:
    - Tuple containing the predicted class and the associated probability.
    """
    # Transform the input document using the trained vectorizer
    doc_vectorized = vectorizer.transform([doc])

    # Get the predicted probabilities for each class
    probas = classifier.predict_proba(doc_vectorized).flatten()

    # Identify the class with the highest probability
    max_proba_idx = probas.argmax()

    if labels:
        most_proba_class = labels[max_proba_idx]
    else:
        most_proba_class = max_proba_idx

    return most_proba_class, probas[max_proba_idx]

if __name__ == "__main__":
    # Example usage:
    loaded_nb_classifier, loaded_vectorizer = load_model_and_vectorizer()

    training_corpus = fetch_20newsgroups(subset='train')
    print(training_corpus.target_names)
    # Classify a new text
    s = "New Toyota 86 Launch Reportedly Delayed to 2022, CEO Doesn't Want a Subaru Copy"
    classes = ['Return Policy', 'Refund Policy','Privacy Policy','Payment Policy','Delivery Policy','Other Document']
    answer = classify_text(loaded_nb_classifier, loaded_vectorizer, s, training_corpus.target_names)

    # Display the result
    print("Predicted class:", answer[0])
    print("Probability:", answer[1])
