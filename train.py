# train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Iris data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Train a RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, preds):.2f}")

# 5. Save model to disk
joblib.dump(model, "model.pkl")
print("Saved model to model.pkl")
