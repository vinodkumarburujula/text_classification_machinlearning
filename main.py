
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("news_dataset.csv")

# 2. Validate and clean dataset
df = df.dropna(subset=["text", "category"])
df["text"] = df["text"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df = df[(df["text"] != "") & (df["category"] != "")]
df = df.drop_duplicates(subset=["text"])

# 3. Separate input and output
X = df["text"]
y = df["category"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# 6. Train the model
model.fit(X_train, y_train)

joblib.dump(model, "news_classifier.joblib")
print("Model saved successfully!")

# 7. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Predict a new news article
new_text = [
    "The cricket team won the championship after a great match."
]

prediction = model.predict(new_text)

print("\nNew Text:", new_text[0])
print("Predicted Category:", prediction[0])
