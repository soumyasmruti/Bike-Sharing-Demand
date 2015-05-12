import pandas as pd
import numpy as np
from sklearn import ensemble
from datetime import datetime
import math
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV


date_str = '%Y-%m-%d %H:%M:%S'

features = ['season', 'holiday', 'workingday', 'weather',
        'temp', 'atemp', 'humidity', 'windspeed', 'year',
         'month', 'weekday', 'hour', 'day']

def format_date(data):
    #Return Date time tuple
    d = pd.DatetimeIndex(data['datetime'])
    return {'year': d.year, 'month': d.month, 'day': d.day,
            'hour': d.hour, 'weekday': d.weekday()}


def process_new_attributes(data):
    #Adding new date attributes to the frame
    temp = pd.DatetimeIndex(data['datetime'])
    data['day'] = temp.day
    data['year'] = temp.year
    data['month'] = temp.month
    data['hour'] = temp.hour
    data['weekday'] = temp.weekday
        
    return data

def load_data():
    train = pd.read_csv("train.csv")
    train = process_new_attributes(train)
    for col in ['casual', 'registered', 'count']:
        train['log-' + col] = train[col].apply(lambda x: math.log(1 + x))
    
    y_train = np.asarray(train.count) 
    X_train = train.drop(["count"],axis =1)
    
    
    test = pd.read_csv("test.csv")
    test = process_new_attributes(test)
    #X_test = test.drop(["datetime"],axis =1)
        
    return X_train, y_train, test

def merge_predict(model1, model2, test_data):
#    Combine the predictions of two separately trained models.
#    The input models are in the log domain and returns the predictions
#    in original domain.
    p1 = np.expm1(model1.predict(test_data))
    p2 = np.expm1(model2.predict(test_data))
    #p3 = np.expm1(model3.predict(test_data))
    #p4 = np.expm1(model4.predict(test_data))
    p_total = (p1+p2)
    return(p_total)
     
            
def train_and_predict():
    print "training..."
    training, y_train, X_test = load_data()
    
    
    
    est_casual = ensemble.RandomForestRegressor(n_estimators=1000, n_jobs=-1, min_samples_split=11, oob_score=True, verbose=1)
    est_registered = ensemble.GradientBoostingRegressor(n_estimators=80, learning_rate = .05)
    param_grid2 = {'max_depth': [6],
                'min_samples_leaf': [3, 5, 10, 20],
                }
    
    gs_casual = GridSearchCV(est_casual, param_grid2, n_jobs=-1).fit(training[features], training['log-casual'])
    gs_registered = GridSearchCV(est_registered, param_grid2, n_jobs=-1).fit(training[features], training['log-registered'])      
    
    result3 = merge_predict(gs_casual,  gs_registered, X_test[features])
    df=pd.DataFrame({'datetime':X_test['datetime'], 'count':result3})
    df.to_csv('output.csv', index = False, columns=['datetime','count']) 
    training = training.drop(["datetime"],axis =1)
    X_test = X_test.drop(["datetime"],axis=1)
    training.to_csv('final_train.csv', sep=',')
    X_test.to_csv('final_test.csv', sep=',')  
    
    
#def gen_submission():
#    #model = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=1) 
#    #model = ensemble.ExtraTreesClassifier(n_estimators=50, n_jobs=-1, verbose=1)
#    #model = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=10, random_state=0)   

if __name__ == '__main__':
    train_and_predict()
    
    