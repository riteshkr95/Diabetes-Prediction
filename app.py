from flask import Flask ,request ,render_template ,jsonify
import joblib
import pickle
import numpy as np

# load the train model
model=joblib.load("log_reg.pkl")


app=Flask(__name__)

@app.route("/")
def home():
   return render_template("index.html")

@app.route("/predict" ,methods= ["POST"])

def predict():
       


  
       a=eval(request.form.get("Pregnancies"))
       b=eval(request.form.get("Glucose"))
       c=eval(request.form.get("BloodPressure"))
       d=eval(request.form.get("SkinThickness"))
       e=eval(request.form.get("Insulin"))
       f=eval(request.form.get("BMI"))
       g=eval(request.form.get("DiabetespedigreeFunction"))
       h=eval(request.form.get("Age"))
    


  # make prediction

       prediction=model.predict([[a,b,c,d,e,f,g,h]])

       if prediction[0]== 0 :
            return render_template("P_0.html")
       else:
            return render_template("P_1.html")

       
if __name__ == "__main__" :
     
     app.run(debug=True  ,port=6767 ,host="0.0.0.0")
    

  
  


