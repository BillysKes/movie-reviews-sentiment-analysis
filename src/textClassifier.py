import nltk
from nltk.corpus import movie_reviews
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm


def preprocess_text(text):
    if isinstance(text, list):  # Check if it's a list
        text = " ".join(text)  # Convert the list to a string
    try:
        stop_words = set(stopwords.words("english"))  # a list with the most common english stop words
        words = word_tokenize(text)
        # Create an empty list to store filtered words
        filtered_words = []
        for word in words:
            if word.isalnum() and word.lower() not in stop_words:
                filtered_words.append(word.lower())
        return " ".join(filtered_words)
    except Exception as e:
        print(f"Error processing text: {text}")
        print(str(e))
        return ""


nltk.download('stopwords')
#nltk.download('punkt')
# Load the movie_reviews dataset
nltk.download("movie_reviews")

# Get movie review file IDs, for example : '[neg,pos]/filename.txt'
review_ids = movie_reviews.fileids()
documents = []

# Iterate over categories and file IDs to create documents
# categories -> (positive or negative)
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        text = list(movie_reviews.words(fileid))  # contains a list of every word/character that exist in that review
        documents.append((text, category))

# Shuffle the documents to ensure randomness
shuffle(documents)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()
# Fit and transform the text data
X_list = []
for text, _ in documents:  # category is excluded before preprocessing step
    preprocessed_text = preprocess_text(text)
    X_list.append(preprocessed_text)

X = vectorizer.fit_transform(X_list)  # bag of words model


y_list = []
for _, category in documents:
    y_list.append(category)

y = y_list
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)
