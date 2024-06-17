#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import pyarrow.parquet as pq
import sys


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:

year = "2023"
month = "03"

def run():

    year = sys.argv[1]
    month = sys.argv[2]

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    print(f'The month and year of the data set selected are: year::{year} ; month:: {month}')
    print(f'The average duration predicted for the current data set is: {y_pred.mean()}')

    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')

    output_file = f'{year}_{month}_yellow_taxi_predictions.parquet'

    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    parquet_file = pq.ParquetFile(output_file)
    num_rows = parquet_file.metadata.num_rows
    print(f'Number of rows: {num_rows}')


if __name__ == '__main__':
    run()




