from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os
import json
from prophet import Prophet
from groq import Groq
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
from ahocorasick import Automaton
load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS
firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
if firebase_credentials:
    cred_dict = json.loads(firebase_credentials)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
else:
    raise ValueError("Firebase credentials are missing!")

# Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Groq API Key is missing!")

client = Groq(api_key=groq_api_key)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
keywords = {
    'groceries': ['supermarket', 'groceries'],
    'dining out': ['restaurant', 'dinner', 'lunch'],
    'entertainment': ['cinema', 'movie', 'concert'],
    'basic necessities': ['electric', 'water', 'bill'],
    'transportation': ['bus', 'train', 'taxi'],
    'apparel': ['clothing', 'jeans', 'shirt'],
    'miscellaneous': ['gift', 'miscellaneous']
}
automaton=Automaton()
for category, words in keywords.items():
    for word in words:
        automaton.add_word(word, category)  # Store category with the word

automaton.make_automaton()
def categorize_transaction(row):
    text = f"{str(row['to']).lower()} {str(row['note']).lower()}"
    for _, category in automaton.iter(text):
        return category  # Return first match
    return 'miscellaneous'


def verify_firebase_token():
    auth_header = request.headers.get("Authorization")
    print(auth_header)
    if not auth_header or not auth_header.startswith("Bearer "):
        return None 

    id_token = auth_header.split("Bearer ")[1]
    print(id_token)
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # ✅ Returns user info    
    except Exception as e:
        print("Token verification failed:", str(e))
        return None

@app.route("/upload", methods=["POST"])
def upload_file():
    user = verify_firebase_token()
    print(user)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401 
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    try:
        df = pd.read_csv(file_path)
        df['category'] = None
        df['category'] = df.apply(categorize_transaction, axis=1)
        print(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']) 
        income = df[df['credit/debit'] == 'credit']['amount'].sum()
        print(income)
        ce = df[df['credit/debit'].str.lower() == 'credit']
        df=df[df['credit/debit'].str.lower()=='debit']
        ce['Month'] = ce['date'].dt.to_period('M').astype(str)
        mi=ce.groupby('Month')['amount'].sum().reset_index()
        print(mi)
        print(df)
        print("pp")
        df['Month']=df['date'].dt.to_period('M')
        print(df)
        monthly_expense=df.pivot_table(index='category', columns='Month',values='amount',aggfunc='sum',fill_value=0)

        monthly_expense.columns=monthly_expense.columns.astype(str)
        monthly_expense=monthly_expense.reset_index()
        print(monthly_expense)
        df = monthly_expense 
        df.set_index("category", inplace=True)  
        df1=df
        print(df)
        df = df.T.reset_index()

        df.rename(columns={"index": " "}, inplace=True)
        print(" pp")
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

        predictions = {}

        for category in df.columns[1:]: 
            temp_df = df[['Month', category]].dropna()  
            temp_df.columns = ['ds', 'y']  
            
            if len(temp_df) > 2:
                model = Prophet()
                model.fit(temp_df)

                future = model.make_future_dataframe(periods=1, freq='M')
                forecast = model.predict(future)
                predictions[category] = forecast.iloc[-1]['yhat']  
            else:
                predictions[category] = 0
        print(predictions)
        formatted_data = {key: float(value) for key, value in predictions.items()}
        
        pp=json.dumps(formatted_data, indent=4)
        df1['pred']=formatted_data
        print(df1)
        total_expenses_row = df1.iloc[:, 0:].sum(axis=0)
        print(total_expenses_row)
        print(mi)
        mi.rename(columns={"Month": "ds", "amount": "y"}, inplace=True)
        model = Prophet()
        model.fit(mi)
        future = model.make_future_dataframe(periods=1, freq='M')
        forecast = model.predict(future)
        next_month = forecast.iloc[-1]
        predicted_value = next_month['yhat']
        new_row = pd.DataFrame({"ds": ["pred"], "y": [predicted_value]})
        mi = pd.concat([mi, new_row], ignore_index=True)
        mi.rename(columns={"ds": "Month", "y": "amount"}, inplace=True)
        print(mi)
        print(total_expenses_row)
        ttf = pd.DataFrame(total_expenses_row).T
        ttf=ttf.reset_index(drop=True)
        print(mi.shape)
        print(ttf.shape)
        print(ttf)
        tml = ttf.melt( var_name="Month", value_name="amount")

        print(tml)
        cdf = pd.merge(mi, tml, on="Month", how="right", suffixes=('_actual', '_pred'))
        print(cdf)
        cdf['amount_actual'] = cdf['amount_actual'].fillna(0)  # Replace 'value_actual' with the actual column name
        cdf['savings'] = cdf['amount_actual'] - cdf['amount_pred']
        rdf = cdf[['Month', 'savings']]
        print(rdf)
        print(df1)
        df1 = df1.transpose()
        print(df1)
        df1['savings']=rdf['savings'][:].values
        print(df1)

        df1 = df1.transpose()
        print(df1)



        act=df1.to_json()
        ct=[]
        for category in df1.index:
            ct.append(category)
        print(ct)
        j=0
        exc=[]

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"this is my data answer as if you were talking to me so you have to provide analysis on the set of data and give insights for me for each category by analysing patterns and also provide some recommendations on how to save money. Dont use fancy formatting in your answer use plain text{act}"
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )

        content = chat_completion.choices[0].message.content

        print(content)

       

    #     # print( jsonify({"analysis": response["choices"][0]["message"]["content"]}))
        return jsonify({"act":act,"exc":exc,"gpt":content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/submit', methods=['POST'])
def receive_data():
    user = verify_firebase_token()
    print(user)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401 
    try:
        data = request.get_json()  # Get JSON data from frontend
        print(data)
        # Convert JSON to Pandas DataFrame
        df = pd.DataFrame(data)
        print(df)
        df['category'] = None
        df['category'] = df.apply(categorize_transaction, axis=1)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']) # strings with None
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.rename(columns={"type": "credit/debit"})
        income = df[df['credit/debit'] == 'credit']['amount'].sum()
        print(income)
        print(df)
        ce = df[df['credit/debit'].str.lower() == 'credit']

        df=df[df['credit/debit'].str.lower()=='debit']
        df['Month']=df['date'].dt.to_period('M')
        ce['Month'] = ce['date'].dt.to_period('M').astype(str)
        
        mi=ce.groupby('Month')['amount'].sum().reset_index()
        print(mi)
        print(df)
        print("pp")
        monthly_expense=df.pivot_table(index='category', columns='Month',values='amount',aggfunc='sum',fill_value=0)
        print(df)
        monthly_expense.columns=monthly_expense.columns.astype(str)
        monthly_expense=monthly_expense.reset_index()
        df = monthly_expense 
        df.set_index("category", inplace=True)  
        df1=df
        print(df1)
        print("p")
        df = df.T.reset_index()

        df.rename(columns={"index": " "}, inplace=True)
        # print(df1)
        print(" pp")
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

        predictions = {}

        for category in df.columns[1:]: 
            temp_df = df[['Month', category]].dropna()  
            temp_df.columns = ['ds', 'y']  
            
            if len(temp_df) > 2:
                model = Prophet()
                model.fit(temp_df)

                future = model.make_future_dataframe(periods=1, freq='M')
                forecast = model.predict(future)
                predictions[category] = forecast.iloc[-1]['yhat']  
            else:
                predictions[category] = 0
        print(predictions)
        formatted_data = {key: float(value) for key, value in predictions.items()}

            # Print in a readable format
        
        pp=json.dumps(formatted_data, indent=4)
        # pp=json.dumps(df1, indent=4)
        df1['pred']=formatted_data
        total_expenses_row = df1.iloc[:, 0:].sum(axis=0)
        print(total_expenses_row)
        print(mi)
        mi.rename(columns={"Month": "ds", "amount": "y"}, inplace=True)
        model = Prophet()
        model.fit(mi)
        future = model.make_future_dataframe(periods=1, freq='M')
        forecast = model.predict(future)
        next_month = forecast.iloc[-1]
        predicted_value = next_month['yhat']
        new_row = pd.DataFrame({"ds": ["pred"], "y": [predicted_value]})
        mi = pd.concat([mi, new_row], ignore_index=True)
        mi.rename(columns={"ds": "Month", "y": "amount"}, inplace=True)
        print(mi)
        print(total_expenses_row)
        ttf = pd.DataFrame(total_expenses_row).T
        ttf=ttf.reset_index(drop=True)
        print(mi.shape)
        print(ttf.shape)
        print(ttf)
        tml = ttf.melt( var_name="Month", value_name="amount")

        print(tml)
        cdf = pd.merge(mi, tml, on="Month", how="right", suffixes=('_actual', '_pred'))
        print(cdf)
        cdf['amount_actual'] = cdf['amount_actual'].fillna(0)  # Replace 'value_actual' with the actual column name
        cdf['savings'] = cdf['amount_actual'] - cdf['amount_pred']
        rdf = cdf[['Month', 'savings']]
        print(rdf)
        print(df1)
        df1 = df1.transpose()
        print(df1)
        df1['savings']=rdf['savings'][:].values
        print(df1)

        df1 = df1.transpose()
        print(df1)
        act=df1.to_json()
        
        j=0
        exc=[]
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze my expense data and provide insights for each category by identifying patterns. Highlight any anomalies and suggest actionable recommendations to help me save money. Keep the response concise and in plain text without fancy formatting.{act}"
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )

        content = chat_completion.choices[0].message.content
        return jsonify({"act":act,"exc":exc,"gpt":content})
        return jsonify({"message": "Data received successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run()
