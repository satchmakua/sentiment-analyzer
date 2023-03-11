import nltk
import random
import sklearn
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('movie_reviews')

# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(document):
    """
    Tokenize and lemmatize the document.
    """
    tokens = word_tokenize(document.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)

# Preprocess the movie_reviews dataset
preprocessed_documents = [(preprocess(" ".join(document)), category) for document, category in documents]

# Split the dataset into training and testing sets
train_size = int(0.2 * len(preprocessed_documents))
train_documents = preprocessed_documents[:train_size]
test_documents = preprocessed_documents[train_size:]

# Vectorize the documents
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([document for document, category in train_documents])
y_train = [category for document, category in train_documents]
X_test = vectorizer.transform([document for document, category in test_documents])
y_test = [category for document, category in test_documents]

# Train the LinearSVC model
clf = LinearSVC(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test the model on a sample review
review = "This movie was fantastic! The acting was superb and the plot was engaging. So amazing, and the actors all did wonderfully!"
preprocessed_review = preprocess(review)
X_review = vectorizer.transform([preprocessed_review])
y_review = clf.predict(X_review)
print("Predicted sentiment:", y_review[0])
