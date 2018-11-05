from flask import Flask
from flask import jsonify,request
from LR import Model1
clf = Model1()

app = Flask(__name__)

@app.route("/read")
def read():
	clf.read_df("C:\\Users\\A7md\\Desktop\\Assigment 3\\diabetes\\diabetes.csv")
	#clf.read_df('C:\\Users\\hebaa\\Desktop\\NTI\\diabetes.csv')
	#C:\Users\A7md\Desktop\Assigment 3\
	return clf.dataset.head().to_json()
	return ("read Done")

@app.route("/split")
def split():
	clf.split_df()
	return "Data split done "

@app.route("/scale")
def scale():
	clf.Scaling()
	return "scalling Done!"

@app.route("/train_test")
def train_test():
	clf.train_test(0.25)
	return "train_test Done !"

@app.route("/train_page")
def train_page():
    return "<form method = 'GET' action ='http://127.0.0.1:9090/train'>\
    <h2>Algorizm:</h2>\
    </br>\
    <select name='modelname'>\
		<option value='LogisticRegression'>LogisticRegression</option>\
		<option value='KNeighborsClassifier'>KNeighborsClassifier</option>\
		<option value='SVC'>SVC</option>\
		<option value='DecisionTreeClassifier'>DecisionTreeClassifier</option>\
		<option value='RandomForestClassifier'>RandomForestClassifier</option>\
    </select>\
  </br></br>\
  <input type='submit' value='train'>\
  </form>"

@app.route("/train")
def train():
	model_name = request.args.get('Modelname')
	clf.train(model_name)
	return "training Done !"



@app.route("/evaluate")
def evaluate():
	score = clf.evaluate()
	resp = {"score" : score}
	return jsonify(resp)

@app.route("/predict", methods = ["GET"])
def predict():
	Pregnancies	=  request.args.get('Pregnancies')
	Glucose =  request.args.get('Glucose')
	BloodPressure =  request.args.get('BloodPressure') 	
	SkinThickness	=  request.args.get('SkinThickness')
	Insulin	=  request.args.get('Insulin')
	BMI	=  request.args.get('BMI')
	DiabetesPedigreeFunction	=  request.args.get('DiabetesPedigreeFunction')
	Age =  request.args.get('Age')
	y_pred = clf.predict([[Pregnancies,Glucose, BloodPressure,SkinThickness,Insulin , BMI,DiabetesPedigreeFunction,Age]])
	#print ("=================  " ,int(y_pred[0]))
	resp = {"class" : int(y_pred[0])}
	return jsonify(resp)


if __name__ == '__main__':
	try:
	    app.run(port ='9090' , host = '127.0.0.1')
	except Exception as e:
		print ("Error")