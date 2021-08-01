
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import sklearn

app = Flask(__name__)
print(sklearn.__version__)
model = pickle.load(open('model\model.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    #print(request.form.values)
    district_code = int(request.form['district_code'])
    #print(district_code)
    neighborhoodcode = int(request.form['neighborhood.code'])
    #print(neighborhoodcode)
    gender = request.form['gender']
    #print(gender)
    age_group = request.form['age_group']
    #print(age_group)
    year = int(request.form['year'])
    #print(year)
    
    datavalues=[[year,district_code,neighborhoodcode,age_group,gender]]
    #print(datavalues)
    data1 = pd.DataFrame(datavalues, columns = ['Year', 'District.Code', 'Neighborhood.Code','Age','Gender'])
    #print(data1)

    data1['Age'].replace(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
       '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74',
       '75-79', '80-84', '85-89', '90-94', '>=95'],[4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,95],inplace = True)
    data1['Gender'].replace(['Female','Male'],[1,0],inplace = True)

    X = data1[['Year','District.Code','Neighborhood.Code','Age','Gender']]
    features = X.iloc[:,:].values
    out = model.predict(features)

    return render_template('result.html',out=(np.round((out[0]))))

if __name__ == "__main__":

    app.run()
   
    
   