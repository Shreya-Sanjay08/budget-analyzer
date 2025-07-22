import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load your labeled expense data
# It should have at least two columns: 'Description' and 'Category' (with values: 'Need' or 'Want')
df = pd.read_csv("expanded_expenses_with_dates.csv")  # Replace with your dataset

# Text and target
X = df["ItemDescription"]
y = df["Label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

# Transform text
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
