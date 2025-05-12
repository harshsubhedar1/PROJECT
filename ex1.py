import joblib

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load new emails (1 per line, no labels)
with open('email.txt', 'r', encoding='utf-8') as file:
    emails = [line.strip() for line in file if line.strip()]

# Transform and predict
email_vectors = vectorizer.transform(emails)
predictions = model.predict(email_vectors)

# Output
print("\n--- Spam Detection Results ---")
for email, prediction in zip(emails, predictions):
    print(f"\nEmail: {email}")
    print("Result:", "Spam" if prediction == 1 else "Not Spam")
    print("-----------------------------------")
