import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the vectorizer and classifier from disk
vectorizer = joblib.load('vectorizer.joblib')
clf = joblib.load('classifier.joblib')

# Load training dataset
train_df = pd.read_csv('train_dataset.csv')

# Load test dataset
test_df = pd.read_csv('test_dataset.csv')

# Convert text data to numerical data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on testing set
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print(classification_report(y_test, y_pred))

# Save the classifier and vectorizer to disk
joblib.dump(clf, 'classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
