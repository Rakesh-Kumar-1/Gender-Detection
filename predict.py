import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def extract_features(name: str):
    """Extract richer features from a given name."""
    name = name.lower().strip()
    vowels = set("aeiou")

    return {
        'first_letter': name[0],
        'last_letter': name[-1],
        'last2_letters': name[-2:] if len(name) > 1 else name,
        'name_length': len(name),
        'vowel_count': sum(1 for ch in name if ch in vowels),
        'ends_with_vowel': name[-1] in vowels
    }


def train_and_evaluate_model(data_path, model_filename, vectorizer_filename):
    """
    Trains a Random Forest model for gender prediction
    and saves the model + vectorizer using pickle.
    """
    if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
        print("Model and vectorizer files already exist. Skipping retraining.")
        return model_filename, vectorizer_filename

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{data_path}' not found.")
        return None, None

    # Extract first names
    data['first_name'] = data['full_name'].apply(lambda x: x.split()[0].strip())

    # Create features and labels
    features = [extract_features(name) for name in data['first_name']]
    labels = data['gender'].tolist()

    # Vectorize features
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features)
    y = labels

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Random Forest trained with accuracy: {accuracy:.2f}")

    # Save model and vectorizer using pickle
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_filename, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("💾 Model and vectorizer saved.")

    return model_filename, vectorizer_filename


def predict_gender(model_filename, vectorizer_filename, name):
    """Predict gender for a given full name."""
    try:
        # Load model and vectorizer using pickle
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_filename, 'rb') as f:
            vectorizer = pickle.load(f)

        first_name_parts = name.split()
        if not first_name_parts:
            print("❌ Invalid name provided.")
            return None

        first_name = first_name_parts[0].strip()
        features = vectorizer.transform([extract_features(first_name)])
        prediction = model.predict(features)
        return prediction[0] 

    except FileNotFoundError:
        print("Prediction failed. Model or vectorizer files not found. Please train the model first.")
        return error("Model or vectorizer files not found.")
    except Exception as e:
        print(f"⚠️ An error occurred during prediction: {e}")
        return error(e)


if __name__ == "__main__":
    dataset_path = "dataset.csv"
    model_file = "gender_predictor_model.pkl"
    vectorizer_file = "gender_predictor_vectorizer.pkl"

    # Train and save the model only if the files don't exist
    model_file, vectorizer_file = train_and_evaluate_model(dataset_path, model_file, vectorizer_file)

    if model_file and vectorizer_file:
        predict_gender(model_file, vectorizer_file, "ShashiKala Devi")
        print("-" * 20)
        predict_gender(model_file, vectorizer_file, "Deepak Priya")