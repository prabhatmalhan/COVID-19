import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(63)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_data = shuffled[:test_set_size]
    train_data = shuffled[test_set_size:]
    return data.iloc[train_data],data.iloc[test_data]

if __name__=='__main__':
    
    #Read Data
    data = pd.read_csv('covid19.csv')
    train , test = data_split(data,.2)
    
    X_train = train[['fever','cough','bodyPain','diarrhea','pneumonia','soreThroat','runnyNose','malaise','sputum','Breatlessness','fatigue']].to_numpy()
    X_test = test[['fever','cough','bodyPain','diarrhea','pneumonia','soreThroat','runnyNose','malaise','sputum','Breatlessness','fatigue']].to_numpy()

    Y_train = train[['probabity']].to_numpy().reshape(2843 ,)
    Y_test = test[['probabity']].to_numpy().reshape(710 ,)

    clf = LogisticRegression()
    clf.fit(X_train,Y_train)

    #opening a file
    file = open('model.pkl','wb')

    #dumping the model to file
    pickle.dump(clf,file)
    file.close()