import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    
    """
    This function loads the data from the database located on the given filepath.
    Input:
    - database_filepath(String): location of database file
    Output:
    - X: message data
    - y: data containing all the category columns for message data
    - category_names: list containing all category names
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("MessageTable", engine)
    df.related.replace(2,1,inplace=True)
    X = df["message"]
    y = df.drop(["message","id", "original", "genre"], axis = 1)
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    This function tokenizes the text.
    Input:
    - text(String): text which has to be tokenized.
    Output:
    - clean_tokens: list of tokens is a result of tokenizing the input text.
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens   


def build_model():
    """
    This function builds the ML model which will be used to train and predict the data.
    Output:
    - cv: ML model which will be used to train and predict the data
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 50, min_samples_split = 2, n_jobs =1)))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split' : [2],
        'clf__estimator__n_jobs' : [1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model and prints the result for the same.
    Input:
    - model: ML model
    - X_test: data containing the messages to test
    - Y_test: data frame containing labeled results for the test data
    - category_names: list of categories
    """
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    This function saves the ml model on the given path.
    Input:
    - model: ML model
    - model_filepath(String): location for saving the model
    """
    
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()