from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# Load and preprocess dataset
car = pd.read_csv('quikr_car.csv')

# Data cleaning and preprocessing
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')

# Reset index after cleaning
car = car.reset_index(drop=True)

# Filter price to remove outliers
car = car[car['Price'] < 6000000]

# Split dataset into features and target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# OneHotEncoding for categorical variables
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# Create column transformer to apply encoding
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and fit the model using pipeline
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

# Save the model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Create the Flask app
app = Flask(__name__)

# Load the cleaned car dataset for dropdown values
cleaned_car_data = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def home():
    # Get unique values for dropdowns
    companies = cleaned_car_data['company'].unique()
    fuel_types = cleaned_car_data['fuel_type'].unique()
    return render_template('index.html', companies=companies, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    name = request.form.get('name')
    company = request.form.get('company')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kms'))
    fuel_type = request.form.get('fuel_type')

    # Create DataFrame for prediction
    sample = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                          data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5))
    
    # Load the model and predict
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    predicted_price = model.predict(sample)
    
    return jsonify(predicted_price=predicted_price[0])  # Return price as JSON

if __name__ == '__main__':
    app.run(debug=True)
