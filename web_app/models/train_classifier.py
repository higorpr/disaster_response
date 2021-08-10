import sys
import nltk
import pandas as pd
import re
import numpy as np
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Loads the dataframe containing the messages and categories from a SQL database and returns the 

        Parameters:
            df (pandas DataFrame): input dataframe of unclean data.

        Returns:
            X (pandas DataFrame) : dataframe of messages (input values).
            Y (pandas DataFrame) : dataframe of dummy values for all categories (targets).
            category_values (str list) : list of all categories.
    '''
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    table = database_filepath.replace('.db','')
    df = pd.read_sql_table('disasterDB',engine)
      
    # defining input and target values for the ML algorith
    X = df.message.values
    Y = df.iloc[:,4:].values
    
    # creating list of category values
    category_values = df.iloc[:,4:].columns
    
    return X, Y, category_values

def tokenize(text):
    '''
    Returns a list with lemmatized tokens from a string.

        Parameters:
            text (str): A string message to be tokenized and lemmatized

        Returns:
            clean_tokens (str list): A list of strings containing the lemmatized tokens.
    '''
    
    # removing URLs from messages in order to reduce machine work
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenizing text
    tokens = word_tokenize(text)
    # Lemmatizing text
    lemmatizer = WordNetLemmatizer()    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok).lower().strip(),pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the machine learning pipeline using grid search to optimize the estimator parameters.

        Parameters:
            text (str): A string message to be tokenized and lemmatized

        Returns:
            cv (GridSeachCV Object): Optimized model of pipeline containing a vectorizer, Tf-Idf transformer and a classifier.
    '''
    # classifier definitiom
    est = RandomForestClassifier()
    
    # pipeline creation
    pipeline = Pipeline(
        steps = [('vect', CountVectorizer(tokenizer=tokenize)),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultiOutputClassifier(estimator=est)),]
    )
        
    # parameter definition
    parameters = {
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 3],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Presents the precision, recall and f1-score for all categories

        Parameters:
            model (ML model): A string message to be tokenized and lemmatized.
            X_test (pandas DataFrame): Test subset of messages to be target of categorization.
            Y_test (pandas DataFrame): Test subset of target values for the machine learning categorization model.
            category_names (str list): List of all categories.
    ''' 
    # predicting outputs
    ypred = model.predict(X_test)
    
    # displaying overall accuracy
    accuracy = (ypred == Y_test).mean()
    print(f'Overall accuracy: {accuracy}')
    
    # displaying reports for each category
    for order, column in enumerate(category_names):
        print(f'Classification Report for category: {column}.\n')
        precision, recall, fscore, support = precision_recall_fscore_support(
            Y_test[:,order],ypred[:,order],average='weighted'
        )
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {fscore}\n')

def save_model(model, model_filepath):
    '''
    Stores model into pickle file.
    
        Parameters:
            model (ML model): Model to be stored into pickle file.
            model_filepath (str): Path to save pickle file.
    
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    '''
    
    Build, trains and evaluates ML model, saving it into a pickle file.
    
    '''
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