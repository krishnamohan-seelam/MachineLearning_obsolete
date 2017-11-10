import os, sys
lib_path = r'F:\MachineLearning'
sys.path.append(lib_path)
 
from  mltools.mlcommon  import load_data,print_dataset_info,split_dataset,auto_scatter_simple

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union
import numpy as  np  
import pandas as pd
#import seaborn as sns
from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import RFE
import statsmodels.api as sm
 
DATASOURCE_PATH = r'F:\DataSource'
FOLDER=r'\bank-additional'
FILE = r'\bank-additional.csv'
filename =DATASOURCE_PATH+FOLDER+FILE


### 'contact_num',,'poutcome_num'
#categorical_vars =['job','marital','education','default','housing','loan','contact','day','month','duration','campaign',
# 'pdays','previous','poutcome']


'''
all_x_cols =["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]
training_cols =['age','job_num', 'martial_num', 'education_num' ,'default_bin','balance', 'housing_bin' ,'loan_bin' ]

    #logreg_rawdataset['y']=(logreg_rawdataset['subscribed_term_deposit']=='yes').astype(int)
    
    ##print(logreg_rawdataset['subscribed_term_deposit'].value_counts())
    ##print(logreg_rawdataset.groupby('subscribed_term_deposit').mean())
    #***print(logreg_rawdataset.index)
    #***print(logreg_rawdataset.columns)

    ##plot_cols =auto_scatter_simple(logreg_rawdataset,all_x_cols,[target_cols],filename)
    
    
 
    ###print(rfe.support_)
    ###print(rfe.ranking_)
    print(data_dummies_df.shape)
    print("RFE ranking")
    print(rfe.ranking_.shape)
    print("RFE support")
    print(rfe.support_.shape)
    
'''

target_cols = "subscribed_term_deposit"
def start_regression():
    
 
    logreg_rawdataset  = load_data(filename,";") 
    ## split categorical and continous variables
    categorical_vars = logreg_rawdataset.describe(include=["object"]).columns
    #print(categorical_vars)
    continuous_vars = logreg_rawdataset.describe().columns

    ## define X,y and categorical data
    XData =logreg_rawdataset[categorical_vars]
    CData =logreg_rawdataset[continuous_vars]
    yData =logreg_rawdataset[target_cols]
  
   
   
    ## create dummies for categorical variables
    data_dummies_df = pd.get_dummies(XData, columns=categorical_vars)
    subscribed = data_dummies_df.subscribed_term_deposit_yes

    ## prepare data and split into train , test sets
    data_dummies_df = data_dummies_df.drop("subscribed_term_deposit_yes", axis=1)
    data_dummies_df = data_dummies_df.drop("subscribed_term_deposit_no", axis=1)
    data_dummies_df = CData.join(data_dummies_df)
    #print(data_dummies_df.columns)
    
    #logistic_regression_model = LogisticRegression(penalty='l1', C=1.0)
    logistic_regression_model = LogisticRegression()
    selected_X = select_features(logistic_regression_model,data_dummies_df,subscribed)
    print(selected_X.values)
    (train_X, test_X ,train_y ,test_y)=split_dataset(data_dummies_df[selected_X.values],subscribed,0.2,1)
    #stats_summary(data_dummies_df[selected_X.values],subscribed)
    print ("train_x size :: ", train_X.shape)
    print ("train_y size :: ", train_y.shape)
    print ("test_x size :: ", test_X.shape)
    print ("test_y size :: ", test_y.shape)

    train_test_model(logistic_regression_model,train_X, test_X ,train_y ,test_y)

       

def stats_summary(X,Y):
    logit_model=sm.Logit(Y,X)
    result=logit_model.fit()
    print (result.summary())

def select_features(logistic_regression_model,data_dummies_df,subscribed):
    rfe = RFE(logistic_regression_model, 12)
    rfe = rfe.fit(data_dummies_df,subscribed)
    
    
    
    final_X =pd.DataFrame({'x':data_dummies_df.columns,'rank':rfe.ranking_})
    selected_X=final_X.query('rank == 1')['x'].astype(str)
    print(rfe.ranking_)
    ######data_dummies_df['rank'] =rfe.ranking_
    ###print(final_X)
    #print(selected_X.values)
    return  selected_X 

def train_test_model(logistic_regression_model,train_X, test_X ,train_y ,test_y):

     ## create model and train 
   
    logistic_regression_model.fit(train_X, train_y)
    predictions =logistic_regression_model.predict(test_X)
    
    train_scores   = cross_val_score(logistic_regression_model, train_X, train_y, cv=3)
    test_scores    = cross_val_score(logistic_regression_model, test_X, test_y, cv=3)
    ## accuracy
    print('train_scores Accuracies: %s' % train_scores )
    print('train_scores Mean accuracy: %s' % np.mean(train_scores))
    print('test_scores Accuracies: %s' % test_scores )
    print('test_scores Mean accuracy: %s' % np.mean(test_scores))
    ##print(predictions[0:5])
    ##print(test_y[0:5])
    print('lr intercept: {0}'.format(logistic_regression_model.intercept_))
    print('lr coefficents: {0}'.format(logistic_regression_model.coef_))
    print (confusion_matrix(test_y, predictions))

    #train_accuracy = model_accuracy(logistic_regression_model, train_data, train_target)
    #print ("Train Accuracy :: ", train_accuracy) 
     
def create_submission_output(output_df, pipe):
    output = pd.concat([output_df, pd.DataFrame(pipe.predict_proba(holdout_dummies_df)[:, 1])], axis=1)
    output.rename(columns = {0:"subscribed"}, inplace=True)
    output.set_index("ID").to_csv("output.csv")
    return output.head()

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