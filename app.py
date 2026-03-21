import pickle
from flask import  Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

application = Flask(__name__)
app=application

# Get the directory where app.py is located
basedir = os.path.dirname(os.path.abspath(__file__))

#import ridge regressor and standard scaler pickle files
# lasso_model=pickle.load(open(os.path.join(basedir, 'models/Lasso_model.pkl'),'rb'))
lass0=pickle.load(open('Lasso_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    if request.method=='POST':
        Temperature=float(request.form['Temperature'])
        RH=float(request.form['RH'])    
        WS=float(request.form['WS'])
        Rain=float(request.form['Rain'])
        FFMC=float(request.form['FFMC'])
        DMC=float(request.form['DMC'])
        ISI=float(request.form['ISI'])
        Classes=float(request.form['Classes'])
        Region=float(request.form['Region'])

        test = pd.DataFrame([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]], 
                           columns=['Temperature', 'RH', 'WS', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])

        print(test)
        data=scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = lass0.predict(data)
        print(data)
        # new_data = scaler.transform(data)
        # print(new_data)
        # result = ridge_model.predict(new_data)[0]
        return render_template('home.html',result=result)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)