import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)  # __name__ double underscores ke sath likhna hai

# Load training data for feature processing
df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template("home.html", query="")

@app.route("/", methods=['POST'])  # 'methods' sahi likhna hai
def predict():
    input_data = [request.form[f'query{i}'] for i in range(1, 20)]

    # Load trained model
    model = pickle.load(open("model.sav", "rb"))

    # Convert input to dataframe
    new_df = pd.DataFrame([input_data], columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                                 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                                 'PaymentMethod', 'tenure'])

    # Combine with original dataframe to ensure consistent encoding
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Bin tenure column
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    # Drop original tenure
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encoding
    df_encoded = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Take only the last row (user input)
    input_encoded = df_encoded.tail(1)

    # Predict
    single = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[:, 1]

    if single[0] == 1:
        o1 = "This customer is likely to be churned"
    else:
        o1 = "This customer is likely to continue!!"

    o2 = f"Confidence: {round(probability[0] * 100, 2)}%"

    # Render result
    return render_template('home.html',
                           output1=o1,
                           output2=o2,
                           **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)
