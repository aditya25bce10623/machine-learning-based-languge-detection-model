# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib as jbl
import sys 
# 1. Loading Data
file_path = r"D:\vityarthi_ad\dataset.csv"
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: File not found at the specified path:", file_path)
    sys.exit() # Prevents the script from crashing later if 'df' doesn't exist

# 2. Preparing Data
X = df['Text']
y = df['language']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Defining classifier: Vectorizer -> Classifier
mod = Pipeline([
    # Tfidf normalizes the text so long English sentences don't dominate
    ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))), 
    ('classifier', MultinomialNB(alpha=0.5))    
])

# 4. Training the Model
mod.fit(X_train, y_train)

# 5. Evaluating Model Accuracy
ac = mod.score(X_test, y_test)
print(f"\nModel Accuracy: {ac * 100:.2f}%")

# 6. Saving the Trained Model
sp = r"D:\vityarthi_ad\lang_model.pkl" 
jbl.dump(mod, sp)

print("Model successfully saved at:", sp)