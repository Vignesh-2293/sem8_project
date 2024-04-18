from flask import Flask , render_template,request
import pandas as pd
import pickle
import numpy as np
#import os

app = Flask(__name__)

model = pickle.load(open('sales1_prd.pkl','rb'))

@app.route("/")
def home():
    return render_template("home1.html")

@app.route("/predpage")
def predpage():
    return render_template("predict.html")

@app.route('/predict', methods=['GET','POST'])
def pred():
    if request.method == 'POST':
        print('post method called')
        print('Form data:', request.form)

        I_weight = float(request.form['iw'])
        #print(I_weight)

        I_visbility = float(request.form['iv'])
        I_fc = int(request.form['ifc'])
        I_type = int(request.form['it'])
        I_mrp = float(request.form['mrp'])
        outlet_size = int(request.form['os'])
        out_age = int(request.form['oa'])
        outlet_Loc_type = int(request.form['olt'])
        outlet_type = int(request.form['ot'])
        inp_data = (I_weight,I_fc,I_visbility,I_type,I_mrp,outlet_size,outlet_Loc_type,outlet_type,out_age)
        inp_data_arr = np.asarray(inp_data)
        reshaped_inp_arr = inp_data_arr.reshape(1,-1)
        prediction = np.exp(model.predict(reshaped_inp_arr)[0])-1
        res = '{:.2f}'.format(prediction)
        print(res)

        return render_template('res.html',res=res)
    
if __name__ == '__main__':
    app.run(debug=True)