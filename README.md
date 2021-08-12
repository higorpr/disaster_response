### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Script Order](#script)
5. [Observations](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

In order to run the scripts in the app, it is necessary to previously install the following packages:
- json
- plotly
- pandas
- joblib
- flask
- sqlalchemy
- sys
- nltk
- pickle

The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

The motivation for this project was the implementation of a Machine Learning algorith into a Flask interface. 

The Machine Learning(ML) algorithm aims classificate messages that could be found in situations of disasters, so that messages posted on the internet during these events
could be used to inform authorities about relevant situations where people could be in need of help.

The messages are classified into 36 categories based on a database provided by Figure Eight.

## File Descriptions<a name="files"></a>

The files in web_app folder were structured in the following way:

### "data" folder
- process_data.py: Script that contains the ETL (Extract, Transform and Load) pipeline. It receives as arguments, respectively, the messages database csv filepath, the
categories database csv filepath and the filepath of the SQL database to be created.
- disaster_messages.csv: File in csv format that contains the database of messages provided by Figure Eight.
- disaster_categories.csv: File in csv format that contains the database of categories provided by Figure Eight.

### "models" folder:
- train_classifier.py: Script that contains the ML pipeline. It loads the SQL database created by "process_data.py", builds, trains and prints an evaluation of the ML model 
based on precision, recall and f1 score. Finally, it saves the model in a pickle file. As arguments, it receives the filepath of the input SQL database and the desired
filepath of the model pickle file.
- tokenizer.py: Module for the tokenizer function used on "train_classifier.py".

### "templates" folder: 
- master.html: Front page for the web app.
- go.html: Result page for the categorization of messages.

### root (web_app):
- run.py: Python script that loads the SQL database and the ML model, generates the plots, applies the ML model to a query provided on the front page of the app and presents
these results in their respective pages on the web app. 
- tokenizer.py: Same file present on the "models" folder.
- Procfile: File that specifies the commands executed by the app on startup.

## Script Order<a name="script"></a>

The order in which one should run the forementioned scripts in order to deploy the web app locally is:

1. process_data.py
2. train_classifier.py
3. run.py

## Observations<a name="results"></a>

The tokenize.py module was inserted on both root and "models" folder as a workaround for an error encountered during the Heroku deploy. This module can be transformed into
a package in the future, avoiding the repeated code on both folders.

The results of the ML classification algorith of messages mainly fall on the "related" category due to the sheer number of entries on the input messages database used to
train the model. In order to improve the ML algorthim training, the number of messages classified as other categories must be improved. That would increase the accuracy of the ML model when
sorting messages that contain different text contexts.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data, without it this project would have not been possible.
