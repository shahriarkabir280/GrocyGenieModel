
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense



# ===============================
# 3. CONSTANTS & ENCODING
# ===============================
REGIONS = ['urban', 'rural']
SEASONS = ['winter', 'spring', 'summer', 'autumn']
EVENTS = ['normal', 'fasting', 'guests', 'sickness', 'travel', 'meal_off']

REGION_MULTIPLIER = {'urban': 1.0, 'rural': 1.1}
SEASON_MULTIPLIER = {'winter': 1.1, 'spring': 1.0, 'summer': 0.9, 'autumn': 1.0}
EVENT_MULTIPLIER = {'normal': 1.0, 'fasting': 0.7, 'guests': 1.3, 'sickness': 0.5, 'travel': 0.3, 'meal_off': 0.2}

BASE_CONSUMPTION = {
    'rice': {'adult_male': 0.3, 'adult_female': 0.25, 'child': 0.15},
    'milk': {'adult_male': 0.2, 'adult_female': 0.18, 'child': 0.3},
    'potato': {'adult_male': 0.25, 'adult_female': 0.2, 'child': 0.15},
    'onion': {'adult_male': 0.1, 'adult_female': 0.1, 'child': 0.05}
}

le_region = LabelEncoder().fit(REGIONS)
le_season = LabelEncoder().fit(SEASONS)
le_event = LabelEncoder().fit(EVENTS)
le_product = None
scaler = None

# ===============================
# 4. UTILITIES & DATA FUNCTIONS
# ===============================
def get_season(date):
    month = date.month
    if month in [12, 1, 2]: return 'winter'
    elif month in [3, 4, 5]: return 'spring'
    elif month in [6, 7, 8]: return 'summer'
    return 'autumn'

def generate_family():
    return {
        'adult_male': random.randint(1, 3),
        'adult_female': random.randint(1, 3),
        'child': random.randint(0, 3)
    }, random.choice(REGIONS)

def calculate_base_consumption(fam, region, season, event, product):
    base = BASE_CONSUMPTION.get(product, {'adult_male': 0.1, 'adult_female': 0.1, 'child': 0.05})
    total = sum(base[k]*fam.get(k, 0) for k in base)
    total *= REGION_MULTIPLIER.get(region, 1.0)
    total *= SEASON_MULTIPLIER.get(season, 1.0)
    total *= EVENT_MULTIPLIER.get(event, 1.0)
    total *= np.random.normal(1, 0.05)
    return max(total, 0.01)

def generate_data(products, families=3, days=180):
    data = []
    for _ in range(families):
        fam, region = generate_family()
        start = datetime.today() - timedelta(days=days)
        for d in range(days):
            date = start + timedelta(days=d)
            season = get_season(date)
            event = random.choices(EVENTS, weights=[70,5,5,5,5,10], k=1)[0]
            for prod in products:
                cons = calculate_base_consumption(fam, region, season, event, prod)
                stock = random.uniform(1.0, 10.0)
                pred_days = stock / cons if cons > 0 else 1
                act_days = pred_days * np.random.normal(1, 0.1)
                finish_error = int(act_days - pred_days)
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product': prod, 'region': region, 'season': season, 'event': event,
                    'adult_male': fam['adult_male'], 'adult_female': fam['adult_female'], 'child': fam['child'],
                    'consumption': cons, 'finish_error': finish_error, 'finish_days': act_days
                })
    return pd.DataFrame(data)

def is_new_product(product, existing_df):
    return product not in existing_df['product'].unique()

def generate_and_append_new_product(product, csv_path="initial_data.csv"):
    print(f"Generating synthetic data for new product: {product}")
    new_df = generate_data([product], families=3, days=180)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"Product '{product}' added to dataset.")

def fetch_feedback_for_user(user_id):
    response = supabase.table("feedback_data").select("*").eq("user_id", user_id).execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()

def insert_feedback(user_id, df):
    df['user_id'] = user_id
    response = supabase.table("feedback_data").insert(df.to_dict(orient="records")).execute()
    print(response)

def store_predictions(user_id, predictions, user_input):
    rows = []
    for product, result in predictions.items():
        row = {
            'user_id': user_id,
            'date': datetime.today().strftime('%Y-%m-%d'),
            'product': product,
            'region': user_input['region'],
            'season': user_input['season'],
            'event': user_input['event'],
            'adult_male': user_input['family']['adult_male'],
            'adult_female': user_input['family']['adult_female'],
            'child': user_input['family']['child'],
            'predicted_consumption': result['predicted_consumption'],
            'predicted_finish_days': result['predicted_finish_days'],
            'predicted_finish_date': result['predicted_finish_date'],
            'predicted_finish_error': result['predicted_finish_error']
        }
        rows.append(row)
    res = supabase.table("prediction_data").insert(rows).execute()
    print("Predictions stored:", res.status_code)

# ===============================
# 5. MODELING
# ===============================
def prepare_data(df, products):
    global le_product, scaler
    le_product = LabelEncoder().fit(products)
    df['region_enc'] = le_region.transform(df['region'])
    df['season_enc'] = le_season.transform(df['season'])
    df['event_enc'] = le_event.transform(df['event'])
    df['product_enc'] = le_product.transform(df['product'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    if scaler is None:
        scaler_local = MinMaxScaler()
        df[['adult_male','adult_female','child','consumption']] = scaler_local.fit_transform(df[['adult_male','adult_female','child','consumption']])
    else:
        scaler_local = scaler
        df[['adult_male','adult_female','child','consumption']] = scaler_local.transform(df[['adult_male','adult_female','child','consumption']])
    X, y1, y2, y3 = [], [], [], []
    seq_len = 7
    for p in df['product_enc'].unique():
        sub = df[df['product_enc']==p].reset_index(drop=True)
        feats = sub[['adult_male','adult_female','child','region_enc','season_enc','event_enc']].values
        c = sub['consumption'].values
        err = sub['finish_error'].values
        days = sub['finish_days'].values
        for i in range(len(sub)-seq_len):
            X.append(feats[i:i+seq_len])
            y1.append(c[i+seq_len])
            y2.append(err[i+seq_len])
            y3.append(days[i+seq_len])
    return np.array(X), np.array(y1), np.array(y2), np.array(y3), scaler_local

def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64)(inp)
    x = Dense(32, activation='relu')(x)
    out1 = Dense(1, name='daily_consumption_output')(x)
    out2 = Dense(1, name='finish_error_output')(x)
    out3 = Dense(1, name='finish_days_output')(x)
    model = Model(inputs=inp, outputs=[out1, out2, out3])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

def load_or_train_model():
    global scaler
    if not os.path.exists("initial_data.csv"):
        pd.DataFrame(columns=['date','product','region','season','event','adult_male','adult_female','child','consumption','finish_error','finish_days']).to_csv("initial_data.csv", index=False)
    df = pd.read_csv("initial_data.csv")
    if df.empty:
        df = generate_data(list(BASE_CONSUMPTION.keys()))
        df.to_csv("initial_data.csv", index=False)
    products = df['product'].unique().tolist()
    X, y1, y2, y3, scaler_obj = prepare_data(df, products)
    scaler = scaler_obj
    model_path = "global_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(
            X,
            {'daily_consumption_output': y1,
             'finish_error_output': y2,
             'finish_days_output': y3},
            epochs=10,
            batch_size=32,
            validation_split=0.1,
        )
        model.save(model_path)
    return model

def retrain_model_with_feedback(user_id):
    base_df = pd.read_csv("initial_data.csv")
    feedback_df = fetch_feedback_for_user(user_id)
    if feedback_df.empty:
        print("No feedback found for user:", user_id)
        return
    combined_df = pd.concat([base_df, feedback_df], ignore_index=True)
    X, y1, y2, y3, scaler_obj = prepare_data(combined_df, combined_df['product'].unique().tolist())
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(
        X,
        {'daily_consumption_output': y1,
         'finish_error_output': y2,
         'finish_days_output': y3},
        epochs=10,
        batch_size=32,
        validation_split=0.1,
    )
    model.save("global_model.h5")
    print("Retrained model saved.")

# ===============================
# 6. PREDICTION
# ===============================
def predict_user_input(user_input, model):
    global le_product, scaler
    initial_df = pd.read_csv("initial_data.csv")
    predictions = {}
    for product in user_input['stock'].keys():
        if is_new_product(product, initial_df):
            generate_and_append_new_product(product)
            initial_df = pd.read_csv("initial_data.csv")
        products = initial_df['product'].unique().tolist()
        le_product = LabelEncoder().fit(products)
        vec = []
        for _ in range(7):
            base = calculate_base_consumption(user_input['family'], user_input['region'], user_input['season'], user_input['event'], product)
            raw = [
                user_input['family']['adult_male'],
                user_input['family']['adult_female'],
                user_input['family']['child']
            ]
            df_input = pd.DataFrame([raw + [0]], columns=['adult_male', 'adult_female', 'child', 'consumption'])
            raw_scaled = scaler.transform(df_input)[0][:3]
            region_enc = le_region.transform([user_input['region']])[0]
            season_enc = le_season.transform([user_input['season']])[0]
            event_enc = le_event.transform([user_input['event']])[0]
            features = list(raw_scaled) + [region_enc, season_enc, event_enc]
            vec.append(features)
        vec = np.array(vec)[np.newaxis, :, :]
        y1, y2, y3 = model.predict(vec, verbose=0)
        daily = float(y1[0][0])
        error = float(y2[0][0])
        days = float(y3[0][0])
        finish_date = datetime.today() + timedelta(days=days)
        predictions[product] = {
            'predicted_consumption': round(daily, 3),
            'predicted_finish_days': round(days, 2),
            'predicted_finish_date': finish_date.strftime('%Y-%m-%d'),
            'predicted_finish_error': round(error, 2)
        }
    return predictions

# ===============================
# 7. EXAMPLE RUN
# ===============================
model = load_or_train_model()

user_input = {
    "family": {"adult_male": 2, "adult_female": 2, "child": 1},
    "region": "urban",
    "season": "summer",
    "event": "normal",
    "stock": {"rice": 5, "milk": 3, "chicken": 4}  # 'chicken' is new
}

results = predict_user_input(user_input, model)
print(pd.DataFrame(results).T)

feedback = pd.DataFrame([{
    'date': datetime.today().strftime('%Y-%m-%d'),
    'product': k,
    'region': user_input['region'],
    'season': user_input['season'],
    'event': user_input['event'],
    'adult_male': user_input['family']['adult_male'],
    'adult_female': user_input['family']['adult_female'],
    'child': user_input['family']['child'],
    'consumption': v['predicted_consumption'],
    'finish_error': v['predicted_finish_error'],
    'finish_days': v['predicted_finish_days']
} for k, v in results.items()])

insert_feedback("user001", feedback)
store_predictions("user001", results, user_input)
