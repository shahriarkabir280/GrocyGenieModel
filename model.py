from supabase_client import supabase
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import uuid
import joblib
from pathlib import Path

# 3. CONSTANTS & ENCODING

MODEL_PATH = "/tmp/global_model.h5"
DATA_PATH = "/tmp/initial_data.csv"
SCALER_PATH = "/tmp/scaler.pkl"


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
    'onion': {'adult_male': 0.1, 'adult_female': 0.1, 'child': 0.05},
    'lentils': {'adult_male': 0.15, 'adult_female': 0.12, 'child': 0.08},
    'flour': {'adult_male': 0.2, 'adult_female': 0.18, 'child': 0.1},
    'tea': {'adult_male': 0.01, 'adult_female': 0.01, 'child': 0.005},
    'coffee': {'adult_male': 0.02, 'adult_female': 0.02, 'child': 0.0},
    'almond': {'adult_male': 0.03, 'adult_female': 0.03, 'child': 0.01},
    'sugar': {'adult_male': 0.05, 'adult_female': 0.05, 'child': 0.03},
}

UNIT_CONVERSION_FACTORS = {
    'kg': 1.0, 'kilogram': 1.0, 'kilograms': 1.0,
    'g': 0.001, 'gram': 0.001, 'grams': 0.001,
    'l': 1.0, 'litre': 1.0, 'litres': 1.0, 'lt': 1.0,
    'ml': 0.001, 'millilitre': 0.001, 'millilitres': 0.001,
    'pcs': 1.0, 'piece': 1.0,
}

le_region = LabelEncoder().fit(REGIONS)
le_season = LabelEncoder().fit(SEASONS)
le_event = LabelEncoder().fit(EVENTS)
le_product = None
scaler = None


# 4. UTILITIES & DATA FUNCTIONS

def get_season(dt_obj):
    month = dt_obj.month
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
    base = BASE_CONSUMPTION.get(product, {'adult_male': 0.05, 'adult_female': 0.05, 'child': 0.02})
    total = sum(base.get(k, 0) * fam.get(k, 0) for k in ['adult_male', 'adult_female', 'child'])
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
            date_obj = start + timedelta(days=d)
            season = get_season(date_obj)
            event = random.choices(EVENTS, weights=[70,5,5,5,5,10], k=1)[0]
            for prod in products:
                cons = calculate_base_consumption(fam, region, season, event, prod)
                data.append({
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'product': prod, 'region': region, 'season': season, 'event': event,
                    'adult_male': fam['adult_male'], 'adult_female': fam['adult_female'], 'child': fam['child'],
                    'consumption': cons
                })
    return pd.DataFrame(data)

def is_new_product(product, existing_df):
    return product not in existing_df['product'].unique()

def generate_and_append_new_product(product, csv_path=DATA_PATH):
    print(f"Generating synthetic data for new product: {product}")
    new_df = generate_data([product], families=3, days=180)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"Product '{product}' added to dataset.")

def get_user_details(user_id):
    try:
        response = supabase.table("users").select("*").eq("user_id", user_id).single().execute()
        return response.data
    except Exception as e:
        print(f"Could not fetch user details for {user_id}: {e}")
        return None

def record_actual_finish_date(stock_id: uuid.UUID, actual_finish_date: date):
    try:
        update_result = supabase.table("user_stocks").update({
            "actual_finish_date": actual_finish_date.isoformat(),
            "is_verified": True # Mark as verified for retraining
        }).eq("stock_id", str(stock_id)).execute()
        return bool(update_result.data)
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return False

# 5. MODELING


def prepare_data(df, products, existing_scaler=None):
    global le_product
    le_product = LabelEncoder().fit(products)
    df['region_enc'] = le_region.transform(df['region'])
    df['season_enc'] = le_season.transform(df['season'])
    df['event_enc'] = le_event.transform(df['event'])
    df['product_enc'] = le_product.transform(df['product'])

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    features_to_scale = ['adult_male', 'adult_female', 'child', 'consumption']
    
    if existing_scaler:
        scaler_to_use = existing_scaler
    else:
        scaler_to_use = MinMaxScaler()
        # Fit only on the training data
        df[features_to_scale] = scaler_to_use.fit_transform(df[features_to_scale])
    
    X, y = [], []
    seq_len = 7
    feature_cols = ['adult_male', 'adult_female', 'child', 'region_enc', 'season_enc', 'event_enc', 'product_enc']
    for p in df['product_enc'].unique():
        sub = df[df['product_enc'] == p].reset_index(drop=True)
        # We need to scale the features for the sub-dataframe
        sub[features_to_scale] = scaler_to_use.transform(sub[features_to_scale])
        feats = sub[feature_cols].values
        c = sub['consumption'].values
        
        for i in range(len(sub) - seq_len):
            X.append(feats[i:i + seq_len])
            y.append(c[i + seq_len])
            
    return np.array(X), np.array(y), scaler_to_use


def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = LSTM(32)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, name='daily_consumption_output')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# retrain the model 
def retrain_model_with_feedback(user_id):
    """
    Fetches verified feedback for a user, calculates actual consumption,
    and retrains the global model with this new, high-quality data.
    """
    global ml_model, scaler

    print(f"Starting retraining process for user: {user_id}")

    # 1. Fetch Verified Feedback Data from Supabase
    try:
        feedback_res = supabase.table("user_stocks").select("*").eq("user_id", user_id).eq("is_verified", True).execute()
        if not feedback_res.data:
            return {"success": False, "message": "No new verified feedback found to retrain the model."}
    except Exception as e:
        return {"success": False, "message": f"Database error fetching feedback: {e}"}

    # 2. Get User's Family Details
    user_details = get_user_details(user_id)
    if not user_details:
        return {"success": False, "message": f"Could not find user details for user ID: {user_id}"}
    
    family_data = {
        "adult_male": user_details.get("adult_male", 1),
        "adult_female": user_details.get("adult_female", 1),
        "child": user_details.get("child", 0)
    }
    region = user_details.get("region", "urban")

    # 3. Process Feedback and Create New Training Data
    new_training_data = []
    for record in feedback_res.data:
        try:
            purchase_date_str = record['purchase_date']
            actual_finish_date_str = record['actual_finish_date']

            purchase_date = datetime.fromisoformat(purchase_date_str.replace('Z', '+00:00')).date()
            actual_finish_date = datetime.fromisoformat(actual_finish_date_str.replace('Z', '+00:00')).date()

            duration_days = (actual_finish_date - purchase_date).days
            if duration_days <= 0:
                continue

            unit = (record.get('unit') or 'kg').lower()
            conversion_factor = UNIT_CONVERSION_FACTORS.get(unit, 1.0)
            quantity_in_kg = record['quantity'] * conversion_factor
            
            actual_daily_consumption = quantity_in_kg / duration_days

            # --- THE FIX IS APPLIED HERE ---
            # Generate 3 months of data for each feedback item to give it more weight.
            # This makes the model pay much more attention to the real data.
            for i in range(90): 
                day = purchase_date + timedelta(days=i % duration_days) # Cycle through the actual duration
                new_training_data.append({
                    'date': day.strftime('%Y-%m-%d'),
                    'product': record['product_name'],
                    'region': region,
                    'season': record.get('season') or get_season(day),
                    'event': record.get('household_events') or 'normal',
                    'adult_male': family_data['adult_male'],
                    'adult_female': family_data['adult_female'],
                    'child': family_data['child'],
                    'consumption': actual_daily_consumption
                })
        except (ValueError, TypeError) as e:
            print(f"Skipping record due to invalid date format or null value: {record['stock_id']} - {e}")
            continue


    if not new_training_data:
        return {"success": False, "message": "No valid feedback could be processed."}

    # 4. Combine with Existing Data and Retrain
    new_df = pd.DataFrame(new_training_data)
    
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.drop_duplicates(subset=['date', 'product', 'region', 'season', 'event', 'adult_male', 'adult_female', 'child'], keep='last', inplace=True)
    
    print(f"Retraining model with {len(existing_df)} existing records and {len(new_df)} new (weighted) feedback records.")

    # 5. Re-prepare all data and retrain the model
    products = combined_df['product'].unique().tolist()
    # We must re-fit a new scaler on the combined data, as the new real data might be outside the original scale
    X, y, new_scaler = prepare_data(combined_df, products, existing_scaler=None) 
    
    if X.shape[0] == 0:
       return {"success": False, "message": "Not enough combined data to retrain the model."}
    
    # Increase epochs to let the model learn the new, weighted data more thoroughly.
    ml_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    # 6. Save the new, improved model and scaler
    ml_model.save(MODEL_PATH)
    joblib.dump(new_scaler, SCALER_PATH)

    # Update the global scaler variable for the current session
    scaler = new_scaler
    
    # Mark feedback as used
    stock_ids_used = [str(r['stock_id']) for r in feedback_res.data if r['stock_id']]
    supabase.table("user_stocks").update({"is_verified": False}).in_("stock_id", stock_ids_used).execute()
    
    return {"success": True, "message": f"Model retrained successfully with {len(new_df)} new data points."}

def load_or_train_model():
    global scaler
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        df = generate_data(list(BASE_CONSUMPTION.keys()))
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    products = df['product'].unique().tolist()
    X, y, scaler_obj = prepare_data(df, products)
    
    if X.shape[0] == 0:
       raise ValueError("Not enough data for training.")
       
    scaler = scaler_obj
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=15, batch_size=32, validation_split=0.1, verbose=1)
    
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model/scaler trained and saved.")
    return model

# 6. PREDICTION & RECORDING

def predict_and_record_stock(input_data):
    # This function's logic remains correct and does not need changes.
    # It will now use the retrained model and scaler automatically.
    global le_product, scaler, ml_model

    user_details = get_user_details(input_data.user_id)
    if not user_details:
        raise ValueError(f"User with ID {input_data.user_id} not found.")

    family_data = input_data.family.dict() if input_data.family else {
        "adult_male": user_details.get("adult_male", 1),
        "adult_female": user_details.get("adult_female", 1),
        "child": user_details.get("child", 0)
    }
    
    region = input_data.region or user_details.get("region", "urban")
    event = input_data.event or "normal"
    season = get_season(input_data.purchase_date)

    initial_df = pd.read_csv(DATA_PATH)
    if is_new_product(input_data.product_name, initial_df):
        generate_and_append_new_product(input_data.product_name)
        initial_df = pd.read_csv(DATA_PATH)
    
    products = initial_df['product'].unique().tolist()
    le_product = LabelEncoder().fit(products)

    product_enc = le_product.transform([input_data.product_name])[0]
    
    raw_demographics = [family_data['adult_male'], family_data['adult_female'], family_data['child']]

    base_cons = calculate_base_consumption(
        family_data, region, season, event, input_data.product_name
    )

    df_for_scaling = pd.DataFrame(
        [raw_demographics + [base_cons]],
        columns=['adult_male', 'adult_female', 'child', 'consumption']
    )
    scaled_values = scaler.transform(df_for_scaling)

    region_enc = le_region.transform([region])[0]
    season_enc = le_season.transform([season])[0]
    event_enc = le_event.transform([event])[0]
    
    features = list(scaled_values[0, :3]) + [region_enc, season_enc, event_enc, product_enc]
    vec = np.array([features] * 7)[np.newaxis, :, :]

    scaled_prediction = ml_model.predict(vec, verbose=0)[0][0]
    dummy_array_for_inverse = np.zeros((1, 4))
    dummy_array_for_inverse[0, 3] = scaled_prediction
    daily_consumption_in_kg = max(scaler.inverse_transform(dummy_array_for_inverse)[0, 3], 0.001)

    unit = (input_data.unit or 'kg').lower()
    conversion_factor = UNIT_CONVERSION_FACTORS.get(unit, 1.0)
    quantity_in_kg = input_data.quantity * conversion_factor
    
    print(f"Unit Conversion: Received {input_data.quantity} {unit}. Calculating with {quantity_in_kg:.3f} kg.")
    print(f"Predicted Daily Consumption for {input_data.product_name}: {daily_consumption_in_kg:.3f} kg/day")

    days_to_finish = quantity_in_kg / daily_consumption_in_kg
    predicted_finish_date = input_data.purchase_date + timedelta(days=days_to_finish)
    
    stock_entry = {
        "user_id": input_data.user_id,
        "product_name": input_data.product_name,
        "unit": input_data.unit,
        "quantity": input_data.quantity,
        "purchase_date": input_data.purchase_date.isoformat(),
        "household_events": event,
        "season": season,
        "predicted_finish_date": predicted_finish_date.isoformat()
    }
    
    stock_insert_result = supabase.table("user_stocks").insert(stock_entry).execute()

    if not stock_insert_result.data:
        raise Exception(f"Failed to insert stock record: {stock_insert_result.error}")

    new_stock_id = stock_insert_result.data[0]['stock_id']
    
    return {
        "stock_id": new_stock_id,
        "predicted_finish_date": predicted_finish_date,
        "product_name": input_data.product_name
    }
    
def recalculate_depletion(input_data):
    # This function's logic is also fine and will use the retrained model.
    global scaler, ml_model
    #... (rest of the function is unchanged and correct)
    family_dict = input_data.family.dict()
    initial_df = pd.read_csv(DATA_PATH)
    if is_new_product(input_data.product_name, initial_df):
        generate_and_append_new_product(input_data.product_name)
        initial_df = pd.read_csv(DATA_PATH)

    current_products_list = initial_df['product'].unique().tolist()
    local_le_product = LabelEncoder().fit(current_products_list)
    
    try:
        product_enc = local_le_product.transform([input_data.product_name])[0]
    except ValueError:
        raise ValueError(f"Could not encode product '{input_data.product_name}'.")
    
    raw_demographics = [family_dict['adult_male'], family_dict['adult_female'], family_dict['child']]

    base_cons = calculate_base_consumption(
        family_dict, input_data.region, input_data.season, input_data.event, input_data.product_name
    )
    
    df_for_scaling = pd.DataFrame(
        [raw_demographics + [base_cons]],
        columns=['adult_male', 'adult_female', 'child', 'consumption']
    )
    scaled_values = scaler.transform(df_for_scaling)
    
    region_enc = le_region.transform([input_data.region])[0]
    season_enc = le_season.transform([input_data.season])[0]
    event_enc = le_event.transform([input_data.event])[0]
    
    features = list(scaled_values[0, :3]) + [region_enc, season_enc, event_enc, product_enc]
    vec = np.array([features] * 7)[np.newaxis, :, :]

    scaled_prediction = ml_model.predict(vec, verbose=0)[0][0]
    dummy_array_for_inverse = np.zeros((1, 4))
    dummy_array_for_inverse[0, 3] = scaled_prediction
    daily_consumption_in_kg = max(scaler.inverse_transform(dummy_array_for_inverse)[0, 3], 0.001)

    unit = (input_data.unit or 'kg').lower()
    conversion_factor = UNIT_CONVERSION_FACTORS.get(unit, 1.0)
    quantity_in_kg = input_data.quantity * conversion_factor
    
    days_to_finish = quantity_in_kg / daily_consumption_in_kg
    predicted_finish_date = datetime.today().date() + timedelta(days=days_to_finish)

    return predicted_finish_date.isoformat()


# 8. GLOBAL LOADING FOR FASTAPI
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        ml_model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Loaded trained model and scaler for prediction.")
    except Exception as e:
        print(f"Error loading model/scaler, retraining... Error: {e}")
        ml_model = load_or_train_model()
else:
    print("Model or scaler not found, initiating training...")
    ml_model = load_or_train_model()
