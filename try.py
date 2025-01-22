from modulus_ml.comparator import ModelComparator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize comparator
comparator = ModelComparator()

# Add models
comparator.add_model('logistic', LogisticRegression())
comparator.add_model('random_forest', RandomForestClassifier())

# Run comparison
results = comparator.compare(X, y)

# Print summary
print(comparator.summary())

# Get best model
best_model = comparator.get_best_model()
print(f"Best model: {best_model}")
