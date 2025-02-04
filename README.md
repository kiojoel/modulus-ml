# Modulus-ML

**Modulus-ML** is a Python library for comparing and evaluating machine learning models. It simplifies the process of benchmarking multiple models by providing an easy-to-use interface for model comparison, performance evaluation, and summary generation.

## Features

- Add multiple machine learning models for comparison.
- Evaluate models on classification and regression tasks.
- Generate detailed summaries of model performance.
- Select the best model based on evaluation metrics.

## Installation

To install Modulus-ML, use pip:

```bash
pip install modulus-ml
```

## Usage

Here is a quick example of how to use Modulus-ML:

```python
from modulus_ml import ModelComparator
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
```

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Thanks to the open-source community for providing tools and inspiration for this library.
