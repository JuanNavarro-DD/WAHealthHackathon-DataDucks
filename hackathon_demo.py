import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from time import sleep
from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import json


def predict(day, month, hour, hospital):
    model = load_model('models/best_model.h5')

    with open('train_columns.json') as f:
        train_columns = json.load(f)

    # Sample new data
    df_new = pd.DataFrame({
        'day_of_week': [day]*5,
        'month': [month]*5,
        'hour': [hour]*5,
        'establishment_code': [hospital]*5,
        'triage_category': [1, 2, 3, 4, 5]
    })

    # Apply the same preprocessing steps (e.g., one-hot encoding, scaling, etc.)
    # NOTE: Use the same scaler and encoders you used for the training data.
    df_new = pd.get_dummies(df_new, columns=['day_of_week', 'month', 'hour', 'establishment_code', 'triage_category'])

    df_new_aligned = pd.DataFrame(columns=train_columns)
    for col in df_new.columns:
        df_new_aligned[col] = df_new[col]
    
    df_new_aligned.fillna(False, inplace=True)

    predictions = model.predict(df_new_aligned)

    result = pd.DataFrame(predictions, columns=['count'])
    result['triage_category'] = ['Red','Orange','Yellow','Blue','Green']

    # Return the prediction
    return result[['triage_category', 'count']]

def retrain(day, month, hour, hospital, triage_categoryWithCount: dict):
    model = load_model('models/best_model.h5')

    with open('train_columns.json') as f:
        train_columns = json.load(f)

    # Sample new data
    df_new = pd.DataFrame({
        'day_of_week': [day]*len(triage_categoryWithCount),
        'month': [month]*len(triage_categoryWithCount),
        'hour': [hour]*len(triage_categoryWithCount),
        'establishment_code': [hospital]*len(triage_categoryWithCount),
        'triage_category': triage_categoryWithCount.keys(),
        'count': triage_categoryWithCount.values()
    })

    

    # Apply the same preprocessing steps (e.g., one-hot encoding, scaling, etc.)
    # NOTE: Use the same scaler and encoders you used for the training data.
    x_retrain = df_new.drop(columns=['count'])
    y_retrain = df_new['count']
    x_retrain = pd.get_dummies(x_retrain, columns=['day_of_week', 'month', 'hour', 'establishment_code', 'triage_category'])

    df_new_aligned = pd.DataFrame(columns=train_columns)
    for col in x_retrain.columns:
        df_new_aligned[col] = x_retrain[col]
    
    df_new_aligned.fillna(False, inplace=True)

    new_lr = 0.0001
    optimizer = Adam(learning_rate=new_lr)

    # Recompile the model with the new learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.fit(df_new_aligned, y_retrain, epochs=10, batch_size=5)

def receive_data():
    for i in range(1,79):
        print(f"data received from hospital {i}")
        sleep(0.2)
    pass

if __name__ == '__main__':
    print("Receiving data from the hospitals...")
    receive_data()
    print("Data received!")
    print("Predicting the number of patients per triage level for the next hour...")
    sleep(2)
    print("hospital 7001")
    day = 'Monday'
    month = 'January'
    hour = 8
    hospital = 7001
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("hospital 7002")
    sleep(2)
    day = 'Sunday'
    month = 'May'
    hour = 15
    hospital = 7002
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("hospital 7003")
    sleep(2)
    day = 'Friday'
    month = 'September'
    hour = 13
    hospital = 7003
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("hospital 7017")
    sleep(2)
    day = 'Monday'
    month = 'August'
    hour = 8
    hospital = 7017
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("hospital 7011")
    sleep(2)
    day = 'Monday'
    month = 'September'
    hour = 8
    hospital = 7011
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("hospital 7012")
    sleep(2)
    day = 'Monday'
    month = 'April'
    hour = 13
    hospital = 7012
    result = predict(day, month, hour, hospital)
    print(result)
    sleep(2)
    print("Retraining the model...")
    sleep(2)
    day = 'Monday'
    month = 'January'
    hour = 7
    hospital = 7001
    triage_categoryWithCount = {1: 1, 2: 1, 3: 13, 4: 14, 5: 7}
    retrain(day, month, hour, hospital, triage_categoryWithCount)
    print("Model retrained!")