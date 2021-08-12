### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Setup](#setup)
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

## Setup<a name="setup"></a>

The steps to deploy the web app locally are:

1. Clone the GitHub disaster-response repository (https://github.com/higorpr/disaster_response).
2. Install the packages mentioned on the Installation(#installation) section. If you are using pip, run the command "pip install <package_name>".
3. Go to ../web_app/data (navigate through the folders using "*cd <folder_path>*" on your terminal) and run process_data.py. This can be done by entering the command:
	"*python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db*". This command will create the SQL database DisasterResponse.db on the 
	data folder.
4. Navigate the terminal prompt to ../web_app/models and run train_classifier.py. This can be done by the command: "*python train_classifier.py ../data/DisasterResponse.db*
	*classifier.pkl"* . This command considers that the database file (.db) is in the data folder mentioned in the last step. With that the ML model will be created, trained
	and stored into a pickle file (classifier.pkl). Additionally, you should be able to verify the values of precision, recall and f1-score for all the categories present on
	the file disaster_categories.csv. Be warned that due to teh process of model optimization, this step might take a few minutes.
5. Now, you have all the parts needed to execute run.py in the root folder (web_app). Navigate the terminal prompt to it and run the script by entering the command: "*python*
	*run.py*'. If the the files DisasterResponse.db and classifier.pkl are, respectively, in the data and models folders, there should be no errors. After a successful execution,
	displayed on the terminal, you shall have a url address, where the app will be locally deployed.

Obs.: The app is currently deployed using Heroku, and can be accessed through: https://disaster-higor-app.herokuapp.com/.

## Observations<a name="results"></a>

The tokenize.py module was inserted on both root and "models" folder as a workaround for an error encountered during the Heroku deploy. This module can be transformed into
a package in the future, avoiding the repeated code on both folders.

The results of the ML classification algorith of messages mainly fall on the "related" category due to the sheer number of entries on the input messages database used to
train the model. In order to improve the ML algorthim training, the number of messages classified as other categories must be improved. That would increase the accuracy of the ML model when
sorting messages that contain different text contexts.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data, without it this project would have not been possible.
