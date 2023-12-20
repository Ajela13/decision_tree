import pandas as pd 

df=pd.read_csv("pima-indians-diabetes.csv") 

df.head()
#pregnanr=bumber of times pregnant
#Plasma glucose concentration a 2 hours in an oral glucose tolerance text
#bp=Diastolic blood pressure (mm hg)
#skin=triceps skin fold thickness(mm)
#insulin= 2 hour serum insulin (mu u/ml)
#bmi=body mass index
#pedigree=diabetes pedigree function
#Age=Age in years
#label=class variable(1:tsted positive for diabetes,0: tested negative for diabetes)

feature_cols=['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X=df[feature_cols]
y=df.label

#Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)#70% training and 30% test

#Buinding model without any tuning of hyperparameter
from sklearn.tree import DecisionTreeClassifier

#Create Decision tree classifier object
dtree=DecisionTreeClassifier()

#train decision tree classifier
dtree=dtree.fit(X_train,y_train)


#Predict the response for train and dataset
y_pred_train=dtree.predict(X_train)
y_pred_test=dtree.predict(X_test)

from sklearn.metrics import accuracy_score

#model aaccuracy, how often is the classifier correct?
print(f'Train accuracy {accuracy_score(y_pred_train,y_train)}')
print(f'Test accuracy {accuracy_score(y_pred_test,y_test)}')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

classes=['No Diabetes','Diabetes']

print('TestConfusion matrix')
cf=confusion_matrix(y_pred_test,y_test)
sns.heatmap(cf,annot=True,yticklabels=classes,xticklabels=classes,cmap='Blues',fmt='g')
plt.tight_layout()
plt.show()
cf

#visualizing the graph without the use of graphics
#filled =true filles the color to indicate mayority class
#precision= represents the number of decimal places needed for gini
#rounded= when set to true, draw node boxes with rounded corners
#node:ids= when set to true, show the id number on each node
#proportion=when set to true, change the display of values and/or smples to be proportions and percentages respectively

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(40,40))
dec_tree=plot_tree(decision_tree=dtree,feature_names=X.columns,class_names=['Non-Diabetes','Diabetes'],filled=True,precision=4,rounded=False,node_ids=True,proportion=True)
plt.savefig('one.png')

#visualizing the graph with only depth of "3"

plt.figure(figsize=(40,40))
dec_tree=plot_tree(decision_tree=dtree,feature_names=X.columns,class_names=['Non-Diabetes','Diabetes'],filled=True,precision=4,rounded=False,node_ids=True,proportion=True,max_depth=3,fontsize=22)
plt.savefig('two.png')

from sklearn.model_selection import GridSearchCV
#tunung the parameter using "gridsearchvc"
params={'max_depth':[2,4,6,8,10,12],'min_samples_split':[1,2,3,4],'min_samples_leaf':[2,3,4]}

#total number of trees iterations = 6*3*2

clf=DecisionTreeClassifier(random_state=0)
gcv=GridSearchCV(estimator=clf,param_grid=params)
gcv.fit(X_train,y_train) 

#min_sample_split or min_sample_leaf may or may not come into pictyure depending upon if it is needed or not
#tey runing aboice code with onlu 10 and 12 as depth and see how the result might change

dt_model=gcv.best_estimator_
dt_model

#now using what we have above 


#train decision tree classifier
dt_model.fit(X_train,y_train)


#Predict the response for train and dataset
y_train_pred=dt_model.predict(X_train)
y_test_pred=dt_model.predict(X_test)

from sklearn.metrics import accuracy_score

#model aaccuracy, how often is the classifier correct?
print(f'Train accuracy {accuracy_score(y_train_pred,y_train)}')
print(f'Test accuracy {accuracy_score(y_test_pred,y_test)}')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

classes=['No Diabetes','Diabetes']

print('TestConfusion matrix')
cf=confusion_matrix(y_test_pred,y_test)
sns.heatmap(cf,annot=True,yticklabels=classes,xticklabels=classes,cmap='Blues',fmt='g')
plt.tight_layout()
plt.show()
cf