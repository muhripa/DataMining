from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dan siapkan data
df = pd.read_csv('covid19_sample_data.csv')
features = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
target = ['total_recoveries', 'active_cases']
x = df[features]
y = df[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=70)

# Buat dan latih model regresi
lr = LinearRegression()
lr.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    total_cases = request.form['Total_Cases']
    new_cases = request.form['new_cases']
    total_deaths = request.form['total_deaths']
    new_deaths = request.form['new_deaths']

    # Pastikan input_data sesuai dengan format yang diinginkan
    input_data = np.array([[float(total_cases), float(new_cases), float(total_deaths), float(new_deaths)]])

    # Lakukan prediksi
    prediction = lr.predict(input_data)

    return render_template('index.html', Total_Cases=total_cases, new_cases=new_cases, total_deaths=total_deaths, new_deaths=new_deaths, prediction=prediction[0])
    
if __name__ == '__main__':
    app.run(debug=True)
