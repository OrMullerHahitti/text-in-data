import re
import nltk
from nltk.corpus import wordnet
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)




def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def advanced_preprocess(docs, min_word_length=3, max_word_length=20):
    lemmatizer = nltk.WordNetLemmatizer()

    processed_docs = []
    for doc in docs:
        # Lemmatize with POS tagging
        lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in doc]

        # Filter by word length
        length_filtered = [w for w in lemmatized if min_word_length <= len(w) <= max_word_length]

        processed_docs.append(length_filtered)

    return processed_docs


def remove_rare_frequent_words(docs, rare_threshold=5, frequent_threshold=0.5):
    all_words = [word for doc in docs for word in doc]
    word_counts = Counter(all_words)
    total_docs = len(docs)

    keep_words = set(word for word, count in word_counts.items()
                     if count >= rare_threshold and count / total_docs < frequent_threshold)

    return [[word for word in doc if word in keep_words] for doc in docs]


def remove_low_tfidf_words(docs, tfidf_threshold=0.1, min_df=2, max_df=0.95):
    # Join words back into strings for TfidfVectorizer
    doc_strings = [' '.join(doc) for doc in docs]

    # Initialize TfidfVectorizer with additional parameters
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(doc_strings)
    feature_names = vectorizer.get_feature_names_out()

    # Get mean TF-IDF score for each word
    mean_tfidf = np.mean(tfidf_matrix, axis=0).A1

    # Create a dictionary mapping words to their mean TF-IDF scores
    word_scores = dict(zip(feature_names, mean_tfidf))

    # Filter documents
    filtered_docs = []
    for doc in docs:
        filtered_doc = [word for word in doc if word in word_scores and word_scores[word] >= tfidf_threshold]
        # Ensure the document is not empty after filtering
        if not filtered_doc:
            # If empty, keep the top 3 TF-IDF scoring words from the original document
            doc_scores = [(word, word_scores.get(word, 0)) for word in doc]
            top_words = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:3]
            filtered_doc = [word for word, _ in top_words]
        filtered_docs.append(filtered_doc)

    return filtered_docs