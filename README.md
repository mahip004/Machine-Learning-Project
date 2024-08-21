Text Classifier
This project is a learning project focused on classifying text into one of five categories: Financial, Office & Work, Family and Friends, Advertisement and Promos, and Travel Related.

Table of Contents
Introduction
Features
Installation
Usage
Model Training
Results
Future Work
License
Introduction
The main objective of this project is to build a text classification system capable of categorizing a given text into one of the five specified categories. The system could be potentially used to classify and organize emails, WhatsApp chats, and even books, according to user preferences and priorities, thus saving time and making processes more systematic and efficient.

Features
Text Classification: Categorizes text into one of five predefined categories.
Vectorization: Uses CountVectorizer and TfidfVectorizer to encode the textual data.
Machine Learning Models: Implements RandomForestClassifier and SVC for classification tasks.
Hyperparameter Tuning: Utilizes Grid Search Cross-Validation for model optimization.
Evaluation: Includes confusion matrix visualization for model evaluation.
Model Persistence: Saves the best models and vectorizers using pickle for future use.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/mahi004/text-classifier.git
cd text-classifier
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure you have the required dataset (text_classifier_dataset.csv) in the project directory.

Usage
Run the Python script to train the models:

bash
Copy code
python train_model.py
Test the models with sample sentences:

bash
Copy code
python test_model.py
Model Training
The project involves the following steps:

Data Loading and Preprocessing: The data is loaded and any null values are filled.
Vectorization: Text data is transformed into numerical features using CountVectorizer and TfidfVectorizer.
Model Training: Models are trained using RandomForestClassifier and SVC.
Hyperparameter Tuning: Grid Search Cross-Validation is used for tuning model hyperparameters.
Evaluation: Models are evaluated using accuracy scores and confusion matrices.
Results
Random Forest Classifier: Achieved an accuracy of X% with the best parameters obtained via Grid Search.
Support Vector Machine (SVM): Achieved an accuracy of Y% with the best parameters obtained via Grid Search.
Confusion matrices for both models are plotted to visualize the performance.

Future Work
Additional Vectorization Techniques: Experiment with advanced NLP techniques for better feature extraction.
Deployment: Package the model into a web service for real-time text classification.
Further Optimization: Explore additional algorithms and hyperparameter settings to improve model performance.
On small scale we can classify different sentences among these five fields but on large scae it can be deployed for classifying emails, whatsapp chat and e-books which can save a lot of time and it will be highly beneficial to classify chats and mails according to priorities
