
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
def load_and_preprocess_data():
    df = pd.read_csv('static/arabica_data_cleaned.csv', index_col=0)

    selected_columns = ['Country.of.Origin', 'Region', 'Variety', 'Color', 'Category.One.Defects',
                        'Category.Two.Defects', 'Processing.Method', 'Moisture', "Aroma", "Flavor",
                        "Body", "Sweetness", "Acidity", "Balance", "Uniformity", "Aftertaste"]
    data = df[selected_columns]

    data = data.dropna(how='any')
    data = data[data['Country.of.Origin'] != 'Taiwan'].copy()

    label_encoders = {}
    encoding_dict = {}
    for column in ['Country.of.Origin', 'Region', 'Variety', 'Color', 'Processing.Method']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        encoding_dict[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = data[['Country.of.Origin', 'Region', 'Variety', 'Color', 'Category.One.Defects',
              'Category.Two.Defects', 'Processing.Method', 'Moisture']]
    y = data[["Aroma", "Flavor", "Body", "Sweetness", "Acidity", "Balance", "Uniformity", "Aftertaste"]]


    return X, y, label_encoders
def rf_model(X, y):
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    scaler = MinMaxScaler()

    rmse_list = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_train_normalized = scaler.fit_transform(y_train)
        y_test_normalized = scaler.transform(y_test)

        rf = RandomForestRegressor(n_estimators=23, random_state=42)
        rf.fit(X_train, y_train_normalized)
        y_pred = rf.predict(X_test)

        rmse = mean_squared_error(y_test_normalized, y_pred, squared=False)
        rmse_list.append(rmse)

    print("Average RMSE:", np.mean(rmse_list))
    print("Standard Deviation of RMSE:", np.std(rmse_list))
    return scaler
X, y, label_encoders = load_and_preprocess_data()
scaler = rf_model(X, y)
rf_model_result = load('random_forest_model.joblib')
nn_model_result = load('nearest_neighbors_model.joblib')



def predict_rf_model(rf_model, user_input, scaler):
    prediction = rf_model.predict([user_input])  # No normalization for user_input
    inverse_transformed_prediction = scaler.inverse_transform(prediction)
    return inverse_transformed_prediction

# Encode user input
def rf_model_transform_user_input(user_input, label_encoders):
    transformed_input = []
    label_encoder_keys = list(label_encoders.keys())
    categorical_indices = [0, 1, 2, 3, 6]  # Indices of categorical features

    for i, ui in enumerate(user_input):
        if i in categorical_indices:  # for the categorical features
            transformed_input.append(
                label_encoders[label_encoder_keys[categorical_indices.index(i)]].transform([ui])[0])
        else:  # for the numerical features
            transformed_input.append(ui)

    return transformed_input


def recommend_coffee(user_input, nn_model, label_encoders, X):
    user_input_scaler = MinMaxScaler(feature_range=(0, 1))
    user_input_scaler.fit(np.array([[0], [10]]))
    user_input_normalized = user_input_scaler.transform([user_input])
    distances, indices = nn_model[0].kneighbors(user_input_normalized)
    recommended_coffee = X.iloc[indices[0][0]].copy()

    for column, le in label_encoders.items():
        recommended_coffee[column] = le.inverse_transform([int(recommended_coffee[column])])[0]

    recommended_coffee = pd.DataFrame(recommended_coffee).transpose()
    recommended_coffee = recommended_coffee.drop(['Category.One.Defects', 'Category.Two.Defects'], axis=1)
    return recommended_coffee.squeeze().to_dict()

@app.route('/predict_coffee', methods=['POST'])
def predict():
    user_input = request.get_json(force=True)
    print(user_input)
    transformed_user_input = rf_model_transform_user_input(user_input, label_encoders)
    prediction = predict_rf_model(rf_model_result, transformed_user_input, scaler)
    print(prediction)

    prediction_dict = {"Aroma": round(prediction[0][0], 3),
                       "Flavor": round(prediction[0][1], 3),
                       "Body": round(prediction[0][2], 3),
                       "Sweetness": round(prediction[0][3], 3),
                       "Acidity": round(prediction[0][4], 3),
                       "Balance": round(prediction[0][5], 3),
                       "Uniformity": round(prediction[0][6], 3),
                       "Aftertaste": round(prediction[0][7], 3)}
    print(prediction_dict)
    return jsonify(prediction_dict)

@app.route('/recommend_coffee', methods=['POST'])
def recommend():
    user_input = request.get_json(force=True)
    print(user_input)
    user_input_values = np.array([float(x) for x in user_input])
    recommended_coffee = recommend_coffee(user_input_values, nn_model_result, label_encoders, X)
    print(recommended_coffee)
    return jsonify(recommended_coffee)

if __name__ == "__main__":
    app.run(debug=True)
