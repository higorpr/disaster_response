import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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