import os, sys
lib_path = os.path.abspath(os.path.join('..', '..', '..', 'F:\MachineLearning'))
sys.path.append(lib_path)
 
from  mltools.common  import load_data,print_dataset_info,split_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as  np  
#import seaborn as sns
from sklearn.metrics import roc_curve, auc
DATASOURCE_PATH = r'F:\DataSource'
FOLDER=r'\bank'
FILE = r'\bank-full.csv'
FILENAME =DATASOURCE_PATH+FOLDER+FILE

### 'contact_num',,'poutcome_num'
training_cols =['age','job_num', 'martial_num', 'education_num' ,'default_bin','balance', 'housing_bin' ,'loan_bin' ]
target_cols='subscribed_term_deposit_bin'

def start_regression():
    logreg_rawdataset  = load_data(FILENAME) 
    logreg_dataset =prepare_data(logreg_rawdataset)
    print_dataset_info(logreg_dataset)
    #print(logreg_dataset.head(5))
    XData =logreg_dataset[training_cols]
    yData =logreg_dataset[target_cols]
    (train_data, test_data ,train_target ,test_target)=split_dataset(XData,yData)
    print ("train_x size :: ", train_data.shape)
    print ("train_y size :: ", train_target.shape)
 
    print ("test_x size :: ", test_data.shape)
    print ("test_y size :: ", test_target.shape)
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_data, train_target)
    predictions =logistic_regression_model.predict(test_data)
    
    '''train_accuracy = model_accuracy(logistic_regression_model, train_data, train_target)
    print ("Train Accuracy :: ", train_accuracy)
    '''

    scores   = cross_val_score(logistic_regression_model, train_data, train_target, cv=3)

    print('Accuracies: %s' % scores )
    print('Mean accuracy: %s' % np.mean(scores))
    print(predictions[0:5])
    print(test_target[0:5])
    
    #print('f1 score:', f1_score(test_target, predictions, average="macro"))
    #print('precision score:', precision_score(test_target, predictions, average="macro"))
    #print('recall score:', recall_score(test_target, predictions, average="macro"))   

    precisions = cross_val_score(logistic_regression_model, train_data, train_target, cv=5, scoring='precision')
    print('Precisions: %s' % np.mean(precisions))

    recalls = cross_val_score(logistic_regression_model, train_data, train_target, cv=5, scoring='recall')
    print('Recall: %s' % np.mean(recalls))

    f1s = cross_val_score(logistic_regression_model, train_data, train_target, cv=5, scoring='f1')
    print('f1: %s' % np.mean(f1s))
    print (confusion_matrix(test_target, predictions))

def model_accuracy(trained_model, features, targets, scoring='precision'):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    #accuracy_score = trained_model.score(features, targets)
    accuracy_score  =cross_val_score(features, targets,scoring)
    return accuracy_score

def prepare_data(data_set):
     
    class_le = LabelEncoder()
    class_lb = LabelBinarizer()
    martial_num = class_le.fit_transform(data_set['marital'].values)
    education_num =class_le.fit_transform(data_set['education'].values)
    contact_num  =class_le.fit_transform(data_set['contact'].values)
    poutcome_num =class_le.fit_transform(data_set['poutcome'].values)
    job_num  =class_le.fit_transform(data_set['job'].values)

    default_bin =class_lb.fit_transform(data_set['default'].values)
    housing_bin = class_lb.fit_transform(data_set['housing'].values)
    loan_bin= class_lb.fit_transform(data_set['loan'].values)
    subscribed_term_deposit_bin=class_lb.fit_transform(data_set['subscribed_term_deposit'].values)
    
    data_set['job_num'] =job_num
    data_set['martial_num']=martial_num
    data_set['education_num']=education_num
    data_set['default_bin']=default_bin
    data_set['housing_bin']=housing_bin
    data_set['loan_bin']=loan_bin
    data_set['contact_num']=contact_num
    data_set['poutcome_num']=poutcome_num
    data_set['subscribed_term_deposit_bin']=subscribed_term_deposit_bin
    data_set['duration_num']   = np.where(data_set['duration'] ==0,0,1) 
    
    return data_set 
if __name__ == '__main__':
    #intro_stats()
    start_regression()