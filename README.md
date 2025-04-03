# PP Review App

## Overview
This project is a sentiment analysis application for app reviews. It processes user reviews, extracts key features using Natural Language Processing (NLP) techniques, and applies machine learning models to predict ratings and sentiment polarity (positive or negative sentiment).

## Features
- Data preprocessing: Cleans and normalizes text data.
- Exploratory Data Analysis (EDA): Visualizes rating distributions.
- Bag-of-Words (BoW) feature extraction.
- Logistic Regression models for:
  - Multiclass rating prediction (1-5 stars).
  - Binary sentiment classification (positive/negative).
- Model evaluation with accuracy, confusion matrix, and classification reports.
- Important word coefficient visualization.

## Dataset
The app uses a CSV file (`app_review.csv`) containing app reviews and their corresponding ratings. The dataset is split into training (70%) and testing (30%) sets.

## Installation
### Prerequisites
Ensure you have Python installed along with the necessary libraries:

```bash
pip install pandas numpy matplotlib scikit-learn nltk
```

### Download NLTK Resources
Run the following commands to download required NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
```

## Usage
1. Place your dataset in the project folder as `app_review.csv`.
2. Run the script:

```bash
python review_analysis.py
```
3. Predictions will be saved as `model_predictions.csv`.
4. The script will display:
   - Accuracy metrics
   - Confusion matrices
   - Important word coefficients

## Model Evaluation
The project implements two models:
1. **Logistic Regression for rating prediction** (1 to 5 stars)
2. **Logistic Regression for sentiment classification** (Positive vs. Negative)

Metrics used:
- Accuracy Score
- Precision, Recall, and F1-score
- Confusion Matrix
- Important word coefficient visualization

## Output Files
- `model_predictions.csv`: Contains true vs. predicted ratings and sentiments.
- `important_words_coefficients.png`: Visualization of top influential words.

## Future Improvements
- Implement advanced NLP techniques such as TF-IDF or Word Embeddings.
- Experiment with other classifiers (e.g., Random Forest, SVM, Deep Learning models).
- Improve handling of imbalanced datasets.

## License
This project is open-source and available under the MIT License.

