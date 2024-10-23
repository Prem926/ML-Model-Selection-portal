from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

@app.route('/train', methods=['POST'])
def train_model():
    # Get JSON data from the request
    data = request.get_json()

    # Extract parameters from user input
    model_type = data.get('modelType', 'DecisionTree')
    criterion = data.get('criterion', 'gini')
    max_depth = int(data.get('maxDepth')) if data.get('maxDepth') else None
    min_samples_split = int(data.get('minSamplesSplit', 2))
    min_samples_leaf = int(data.get('minSamplesLeaf', 1))

    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model based on user selection
    if model_type == 'DecisionTree':
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    elif model_type == 'GaussianNB':
        model = GaussianNB()
    elif model_type == 'MultinomialNB':
        model = MultinomialNB()
    elif model_type == 'BernoulliNB':
        model = BernoulliNB()
    else:
        return jsonify({'error': 'Model type not supported'}), 400

    # Train the model and handle any exceptions
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot Decision Tree and encode as Base64 if applicable
    tree_base64 = plot_tree_image(model) if model_type == 'DecisionTree' else None

    # Plot Confusion Matrix and encode as Base64
    conf_matrix_base64 = plot_confusion_matrix(conf_matrix)

    # Return JSON response with accuracy and visualizations
    return jsonify({
        'message': 'Model trained successfully!',
        'accuracy': accuracy,
        'confusionMatrixImage': f'data:image/png;base64,{conf_matrix_base64}',
        'decisionTreeImage': tree_base64,
    })

def plot_tree_image(model):
    """Plot the decision tree and return as Base64."""
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=[f'Feature {i}' for i in range(4)], class_names=['0', '1'])
    tree_img = io.BytesIO()
    plt.savefig(tree_img, format='png')
    tree_img.seek(0)
    return base64.b64encode(tree_img.getvalue()).decode()

def plot_confusion_matrix(conf_matrix):
    """Plot the confusion matrix and return as Base64."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    conf_matrix_img = io.BytesIO()
    plt.savefig(conf_matrix_img, format='png')
    conf_matrix_img.seek(0)
    return base64.b64encode(conf_matrix_img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
