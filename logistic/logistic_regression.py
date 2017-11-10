import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

DATASOURCE_PATH = r'F:\DataSource'
FOLDER=r'\logistic'
FILE = r'\anes_dataset.csv'

FILENAME =DATASOURCE_PATH+FOLDER+FILE

def start_regression():
    logreg_dataset = load_data(FILENAME)

    #print basic features of dataset
    print("No of Observation:",logreg_dataset.shape[0])
    print("No of features:",logreg_dataset.shape[1])
    print("Features",logreg_dataset.columns.values)
    print("Describe dataset",logreg_dataset.describe())

    #split data set into training and test 
    headers = list(logreg_dataset.columns.values)
    print(headers)
    features = headers[:-1]
    target =headers[-1]

    x_train,x_test,y_train,y_test=train_test_split(logreg_dataset[features],logreg_dataset[target],test_size=0.4,
                                                    random_state=1, stratify=logreg_dataset[target])
    
    print ("x_train Shape: ", x_train.shape)
    print ("y_train Shape: ", y_train.shape)
    print ("x_test Shape: ", x_test.shape)
    print ("y_test Shape: ", y_test.shape)
    four_features  = ['TVnews','age','educ','income']

    model_with_4_features = LogisticRegression()
    model_with_4_features.fit(x_train[four_features],y_train)

    train_accuracy = model_with_4_features.score(x_train[four_features], y_train)
    print("Training accuracy",train_accuracy)

    #Logistic Regression
    model_with_all_features = LogisticRegression()
    model_with_all_features.fit(x_train,y_train)
    full_train_accuracy = model_with_all_features.score(x_train, y_train)
    print("full_train_accuracy",full_train_accuracy)

    test_observation1_for_4_features_model = x_test[four_features][:1]

    print("test_observation1_for_4_features_model",model_with_4_features.predict(test_observation1_for_4_features_model))
    test_observation_for_all_features_model = x_test[:1]
    print("test_observation_for_all_features_model",
          model_with_all_features.predict(test_observation_for_all_features_model))

    model_with_4_features_prediction = model_with_4_features.predict(x_test[four_features])
    model_with_4_features_test_accuracy= metrics.accuracy_score(y_test,model_with_4_features_prediction)
    model_with_all_features_prediction = model_with_all_features.predict(x_test)
    model_with_all_features_test_accuracy= metrics.accuracy_score(y_test,model_with_all_features_prediction)

    print ("Model with 4 features test accuracy: ", model_with_4_features_test_accuracy)
    print ("model_with_all_features_test_accuracy: ", model_with_all_features_test_accuracy)

def load_data(filename =DATASOURCE_PATH+FOLDER+FILE):
    logreg_dataset= pd.read_csv(filename)
    return logreg_dataset


if __name__ == '__main__':
    #intro_stats()
    start_regression()
