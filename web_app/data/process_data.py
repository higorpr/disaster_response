import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Returns a pandas DataFrame with the merged messages and categories datasets.

            Parameters:
                messages_filepath (str): Path of the csv file containing the messages.
                categories_filepath (str): Path of the csv file containing the categories.

            Returns:
                df (pandas DataFrame): Pandas DataFrame with messages and categories DataFrames merged.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # removing duplicates of the messages dataset (using 'id' as identifier)
    messages = messages[messages['id'].duplicated() == False]
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # removing duplicates of the categories dataset (using 'id' as identifier)
    categories = categories[categories['id'].duplicated() == False]
    
    # merging datasets
    df = messages.merge(categories,on='id',how='inner')

    return df

def clean_data(df):
    '''
    Returns the cleaned pandas DataFrame containing the messages and the categories devided into columns.

            Parameters:
                df (pandas DataFrame): input dataframe of unclean data.

            Returns:
                final_df (pandas DataFrame): Pandas DataFrame with messages and categories DataFrames merged.
    '''
    # Spliting datasets for easier handling
    categories = df[['id','categories']]
    messages = df[['id', 'message', 'original', 'genre']]

    # dividing into columns the 36 categories of the "categories" dataframe
    categories = categories['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe to be used as column names
    row = categories.iloc[0]
    
    # defining column names for the categories dataset
    category_colnames = row.apply(lambda x: x[:len(x)-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # setting values in categories columns to 0/1(+/-):
        categories[column] = categories[column].apply(lambda x: x[len(x)-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')
        
    # Reindexing the df's for future concatenation:
    categories.index = list(range(len(categories)))
    messages.index = list(range(len(messages)))
    df.index = list(range(len(df)))
    
    # drop the original categories column from the complete dataframe
    df.drop(columns='categories',inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_final = pd.concat([df,categories],axis=1)
    
    # Removing faulty rows (where row related has value 2)
    df_final = df_final[df_final['related'] != 2]
    
    # check and remove duplicates
    df_final = df_final[df_final.duplicated() == False]
    
    return df_final

def save_data(df, database_filename):
    '''
    Saves the dataframe into a SQL database

            Parameters:
                df (pandas DataFrame): input dataframe.
                database_filename (str): desired name of SQL database.

    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disasterDB',engine,if_exists='replace',index=False)

def main():
    '''
    Loads, cleans and saves the message and categories dataframe into a SQL database.
    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()