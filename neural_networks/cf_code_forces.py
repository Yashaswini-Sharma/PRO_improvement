import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('data.csv')

# Step 2: Explore the first few rows to understand its structure
print(df.head())

# Step 3: Handle NaN values in 'problem_tags' column
df['problem_tags'] = df['problem_tags'].fillna('')  # Fill NaN with empty strings
df['problem_tags'] = df['problem_tags'].apply(str)  # Ensure the 'problem_tags' column is string type

# Step 4: Text Processing (Problem Statement)
def clean_text(text):
    if isinstance(text, str):  # Check if the text is a string
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters (except spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    else:
        return ''  # Return an empty string if the value is not a string

# Apply text cleaning to the problem statement
df['cleaned_statement'] = df['problem_statement'].apply(clean_text)

# Step 5: Handle Problem Tags
def extract_rating(tags):
    # Extract rating tags (those starting with *)
    ratings = [tag for tag in tags if tag.startswith('*')]
    # Remove the rating from tags
    tags = [tag for tag in tags if not tag.startswith('*')]
    return tags, ratings

# Convert Problem Tags into lists (assuming tags are stored as comma-separated strings in the CSV)
df['tags'] = df['problem_tags'].apply(lambda x: x.split(','))  # Split tags by commas
df['tags'], df['ratings'] = zip(*df['tags'].apply(extract_rating))

# Step 6: One-Hot Encode the Tags
mlb = MultiLabelBinarizer()
# Fit on the unique tags from the dataset
mlb.fit(df['tags'].explode())  # Use 'explode' to expand lists of tags into individual items
tags_encoded = mlb.transform(df['tags'])

# Convert the encoded tags into a DataFrame
tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

# Step 7: Handle Rating Encoding
# Convert ratings to numerical values (e.g., *1500 -> 1500)
def extract_numerical_rating(rating):
    if rating.startswith('*'):
        return int(rating[1:])
    return 0  # Return 0 for unknown ratings

df['rating_values'] = df['ratings'].apply(lambda x: [extract_numerical_rating(r) for r in x])  # Assign 0 for unknown ratings

# Convert ratings into a single numerical feature (average, sum, etc.)
df['rating_average'] = df['rating_values'].apply(np.mean)

# Step 8: Handle the Label (Difficulty Ranges)
def map_label_to_range(label):
    # Handle cases where the label might be NaN or missing
    if isinstance(label, str) and ' - ' in label:
        label_range = label.split(' - ')
        label_range = [float(x) for x in label_range]
        return np.mean(label_range)  # Using the average of the range
    return np.nan  # Return NaN for invalid or missing labels

# Assuming the 'Label' column contains difficulty range (e.g., "1.00 - 155.80")
df['difficulty_range_avg'] = df['Label'].apply(map_label_to_range)

# Step 9: Combine the cleaned and encoded features into the final DataFrame
df_final = pd.concat([df['problem_name'], df['tags'], tags_df, df['rating_average'], df['difficulty_range_avg']], axis=1)

# Step 10: Display final preprocessed dataframe
print(df_final.head())
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Preprocess the data

# Assuming df_final is already created from previous steps
# Clean the problem statement (ensure it's already clean)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
        return text
    else:
        return ''

df_final['cleaned_statement'] = df_final['problem_statement'].apply(clean_text)

# Step 2: Vectorize the problem statement using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features (adjust as needed)
X_tfidf = tfidf_vectorizer.fit_transform(df_final['cleaned_statement'])

# Step 3: Encode the tags (multi-label)
mlb = MultiLabelBinarizer()
y_tags = mlb.fit_transform(df_final['tags'])

# Step 4: Include the rating as an additional feature (rating_average)
X = np.hstack([X_tfidf.toarray(), df_final['rating_average'].values.reshape(-1, 1)])

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_tags, test_size=0.2, random_state=42)

# Step 6: Train a Multi-Output Classifier (Logistic Regression)
model = MultiOutputClassifier(LogisticRegression(solver='liblinear'))
model.fit(X_train, y_train)

# Step 7: Predict tags on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model (using accuracy, F1-score, and Hamming Loss)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("F1 Score (micro):", f1_score(y_test, y_pred, average='micro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred))

# Step 9: Example of predicting tags for a new problem
new_problem_statement = "You are given a list of integers. Find the maximum sum of a contiguous subarray."
new_rating = 1600  # Example rating
new_tfidf = tfidf_vectorizer.transform([new_problem_statement])
new_features = np.hstack([new_tfidf.toarray(), np.array([[new_rating]])])

# Predict tags
predicted_tags = model.predict(new_features)
predicted_tags = mlb.inverse_transform(predicted_tags)

print(f"Predicted tags: {predicted_tags}")
