from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

#loading the model
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/' , methods=["GET","POST"])

def hello_world():
    if request.method == "POST":
        dict = request.form
        val = list(dict.values())
        # Code for probability
        inputFeatures = list(map(int,val[2:-1]))
        probInfo = clf.predict_proba([inputFeatures])[0][1]
        return render_template('show.html',name=val[0],inf=round(probInfo*100))
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)