from mlsettings.settings import load_app_config, get_datafolder_path
from mltools.mlcommon import load_data, print_dataset_info, split_dataset, auto_scatter_simple

import os
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score,roc_curve, auc

'''
########################################################################################################################
Data Dictionary
########################################################################################################################
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic
parch	# of parents / children aboard the Titanic
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
########################################################################################################################
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
########################################################################################################################
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother(parch=1), father(parch=1)
Child = daughter(parch=2), son(parch=3), stepdaughter(parch=4), stepson(parch=5)
Some children travelled only with a nanny, therefore parch=0 for them.
########################################################################################################################
'''

DIRECTORY = "titanic"
TRAINFILENAME = "train.csv"
TESTFILENAME = "test.csv"
ALL_COLUMNS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
               'Embarked']

X_COLUMNS = ['Fare', 'Pclass', 'Sex_Bin', 'Age', 'SibSp']
Y_COLUMNS = ['Survived']
TRAINING_MODE =True


def start_classification(dc_model ,filename=TRAINFILENAME, is_training=True):
    print("start_classification")
    input_path = get_datafolder_path()
    input_file = os.path.join(input_path, DIRECTORY, filename)

    input_dataset = load_data(input_file)
    print(" input file is :{0} loaded.".format(input_file))
    display_dataset_info(input_dataset)
    # print(input_dataset.columns)

    input_dataset_cleaned = clean_dataset(input_dataset)
    display_dataset_info(input_dataset_cleaned)



    predict_y= None

    if is_training:
        X_values = input_dataset_cleaned[X_COLUMNS]
        y_values = input_dataset_cleaned[Y_COLUMNS]
        dc_model.fit(X_values,y_values)
        acc_decision_tree = round(dc_model.score(X_values, y_values), 4)
        print("Training Accuracy: %0.4f" % (acc_decision_tree))

    else:
        X_values = input_dataset_cleaned[X_COLUMNS]
        out_filename =os.path.join(input_path, DIRECTORY, filename.replace(".csv","_out.csv"))

        print("Predictions are written to  {0}".format(out_filename))
        predict_y = dc_model.predict(X_values)
        input_dataset["Predict_Surived"] =predict_y
        input_dataset.to_csv(out_filename)


        #print(confusion_matrix(y_values, predict_y))
        #print("Precision: %0.4f" % precision_score(y_values, predict_y))

    return True



def learn_model(dc_model,input_dataset_cleaned):
    X_values = input_dataset_cleaned[X_COLUMNS]
    y_values = input_dataset_cleaned[Y_COLUMNS]
    dc_model.fit(X_values,y_values)


def display_dataset_info(input_dataset):
    #print(input_dataset.head(4).to_string())
    print("Data set counts:")
    print(input_dataset.count())
    #print(input_dataset.info())


def clean_dataset(input_dataset):
    columns_to_dropped = ['Cabin', 'Ticket']
    process_dataset = input_dataset.drop(columns_to_dropped, axis=1)
    class_le = LabelEncoder()
    class_lb = LabelBinarizer()
    sex_bin = class_lb.fit_transform(process_dataset['Sex'].values)
    process_dataset['Sex_Bin'] = sex_bin

    mean_age = math.ceil(process_dataset["Age"].mean())
    # print("average age {0}".format(mean_age))
    process_dataset["Age"] = process_dataset["Age"].fillna(mean_age)
    process_dataset["Embarked"] = process_dataset["Embarked"].fillna("NA")
    Embarked_EC = class_le.fit_transform(process_dataset["Embarked"])

    process_dataset["Embarked_EC"] = Embarked_EC

    median_fare = math.ceil(process_dataset["Fare"].median())
    process_dataset["Fare"] = process_dataset["Fare"].fillna(median_fare)
    # print("age {0}".format(process_dataset["Age"].unique()))
    return process_dataset


if __name__ == '__main__':
    load_app_config()
    dc_model = DecisionTreeClassifier(random_state=1)
    start_classification(dc_model,TRAINFILENAME,TRAINING_MODE)
    start_classification(dc_model,TESTFILENAME, not TRAINING_MODE)
