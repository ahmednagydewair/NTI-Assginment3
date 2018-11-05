# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier

class Model1:
	def __init__(self):
		self.sc = StandardScaler()
		self.classifier = LogisticRegression(random_state = 0)

	def	read_df(self ,path):
		self.dataset = pd.read_csv(path)

	def	split_df(self ):
		self.x = self.dataset.iloc[:, :-1]
		self.y = self.dataset.iloc[:, -1]	

	def Scaling(self):
		
		self.x = self.sc.fit_transform(self.x)

	def train_test(self , test1_size):
		
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test1_size, random_state = 0)	

	def train(self,modelname):
		self.read_df("C:\\Users\\A7md\\Desktop\\Assigment 3\\diabetes\\diabetes.csv")
		self.split_df()
		self.Scaling()
		self.train_test(0.25)

		if modelname == "LogisticRegression":
            self.classifier = LogisticRegression()
        elif modelname == "KNeighborsClassifier":
            self.classifier = KNeighborsClassifier(n_neighbors=3)
        elif modelname == "DecisionTreeClassifier":
            self.classifier = DecisionTreeClassifier()
        elif modelname == "RandomForestClassifier":
            self.classifier = RandomForestClassifier( n_estimators=5)
        self.classifier.fit(self.x_train,self.y_train)

	def predict_model(self):
		self.y_pred =  self.classifier.predict(self.x_test)
		return self.y_pred
		
	def evaluate(self):
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score,f1_score
	 	print(classification_report(self.y_test, self.y_pred))
	 	fi_score = f1_score(y_true, y_pred, average='weighted')
		accuracy_score = accuracy_score(y_true, y_pred)
		return self.fi_score , self.accuracy_score

	def predict(self, test2):
		return self.classifier.predict(test2)		
