import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix
)
import nltk
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')

file_path = "app_review.csv"
df = pd.read_csv(file_path)

num_records = df.shape[0]
print('Number of records in the data set:', num_records)

distinct_classes = df['Rating'].nunique()
print('Number of distinct classes in the data set:', distinct_classes)

reviews_with_rating_1 = df[df['Rating'] == 1].sample(5)
reviews_with_rating_5 = df[df['Rating'] == 5].sample(5)

print("5 random reviews with rating 1:")
print(reviews_with_rating_1[['Review', 'Rating']])

print("\n5 random reviews with rating 5:")
print(reviews_with_rating_5[['Review', 'Rating']])

rating_counts = df['Rating'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Reviews by Rating')
plt.xticks(rating_counts.index)
plt.show()

print("Rating distribution:\n", rating_counts)
# The class distribution is imbalanced, with positive reviews (ratings of 4 or 5) significantly outnumbering negative reviews (ratings of 1, 2, or 3).

# Data processing
import re

def clean_data(text):
    if isinstance(text, str):  # Check if input is a string
        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
    else:
        return ""

df['Review'] = df['Review'].apply(clean_data)

print(df[['Review']].head(13))
print(df.loc[7058, "Review"])

random_state = np.random.RandomState(0)
x = df['Review']
y = df['Rating']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

stop_words = stopwords.words('english')

vectorizer = CountVectorizer(stop_words=stop_words, min_df=0.01)
vectorizer.fit(x_train)

X_train_BOW = vectorizer.transform(x_train)
X_test_BOW = vectorizer.transform(x_test)

train_data_BOW = pd.DataFrame(data=X_train_BOW.toarray(), columns=vectorizer.get_feature_names_out())
test_data_BOW = pd.DataFrame(data=X_test_BOW.toarray(), columns=vectorizer.get_feature_names_out())
train_data_BOW.head(5)

# Model Training
model1 = LogisticRegression(random_state=42, max_iter=1000)
model1.fit(train_data_BOW, y_train)

Y_pred1 = model1.predict(test_data_BOW)
accuracy1 = accuracy_score(y_test, Y_pred1)

print("Accuracy of Logistic Regression model1 (BoW):", accuracy1)

import copy
def convert_to_binary_labels(ratings):
    return [1 if rating >= 4 else -1 for rating in ratings]

y_train_binary = copy.deepcopy(y_train)
y_test_binary = copy.deepcopy(y_test)

y_train_binary = convert_to_binary_labels(y_train_binary)
y_test_binary = convert_to_binary_labels(y_test_binary)

print("y_train_binary:", y_train_binary[:10])
print("y_test_binary:", y_test_binary[:10])

model2 = LogisticRegression(random_state=42, max_iter=1000)
model2.fit(train_data_BOW, y_train_binary)

y_pred2 = model2.predict(test_data_BOW)

accuracy2 = accuracy_score(y_test_binary, y_pred2)
print("Accuracy of Logistic Regression model2 (Binary Sentiment):", accuracy2)
print("Classification Report for Binary Sentiment Model:\n", classification_report(y_test_binary, y_pred2))

predictions_model1 = model1.predict(test_data_BOW)

predictions_model2 = model2.predict(test_data_BOW)

predictions_df = pd.DataFrame({
    'True_Rating': y_test,
    'Predicted_Rating': predictions_model1,
    'True_Sentiment': y_test_binary,
    'Predicted_Sentiment': predictions_model2
})

predictions_df.to_csv('model_predictions.csv', index=False)

print("Predictions have been saved to 'model_predictions.csv'.")

#Model Evaluation
accuracy_model1 = accuracy_score(y_test, predictions_model1)
print(f"Accuracy of Model 1 (Rating Prediction): {accuracy_model1:.4f}")

accuracy_model2 = accuracy_score(y_test_binary, predictions_model2)
print(f"Accuracy of Model 2 (Binary Sentiment Prediction): {accuracy_model2:.4f}")

conf_matrix = confusion_matrix(y_test, predictions_model1)
print("Confusion Matrix for Model 1 (Rating Prediction):")
print(conf_matrix)

class_report = classification_report(y_test, predictions_model1, target_names=['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5'])
print("\nClassification Report for Model 1 (Rating Prediction):")
print(class_report)

# Model Analysis
def plot_coefficients(classifier, feature_names, top_features=10, save_plot=False):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    bars = plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=90, ha='right')
    plt.xlabel("Important Words")
    plt.ylabel("Model Coefficient")
    plt.title("Important Words with their Model Coefficients")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_plot:
        plt.savefig("important_words_coefficients.png")
    plt.show()

plot_coefficients(model2, feature_names=vectorizer.get_feature_names_out(), top_features=10)





































