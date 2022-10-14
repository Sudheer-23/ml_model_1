import pickle
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_output():
    if request.method == 'POST':
        f = request.form.get('temp')
        f = float(f)
        prediction = model.predict([[f]])
        output = round(prediction[0],2)
        return render_template('output.html',amt=f'Total Revenue Generated is Rs: {output}/-')
    else:
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(port=4000,debug=True)
