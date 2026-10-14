
import joblib

# Load the saved model
model = joblib.load("news_classifier.joblib")

print("=== News Text Classification ===")
print("Categories: Sports, Technology, Business, Politics")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter a news sentence: ").strip()

    if text.lower() == "exit":
        print("Prediction program stopped.")
        break

    if not text:
        print("Please enter some text.\n")
        continue

    # Predict the category
    prediction = model.predict([text])[0]

    print("Predicted Category:", prediction)
    print("-" * 40)
