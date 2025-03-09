import pandas as pd
import timeit
from os import path

def read_csv():
    pd.read_csv('D:\\Pz2_Data\\train.csv')

def read_parquet():
    pd.read_parquet('D:\\Pz2_Data\\train.parquet')


def convert_to_parquet():
    data = pd.read_csv('D:\\Pz2_Data\\train.csv')
    data.to_parquet('D:\\Pz2_Data\\train.parquet', index=False)

def main():
    csv_start = timeit.default_timer()
    read_csv()
    csv_end = timeit.default_timer()
    print(f"Reading {path.getsize('D:\\Pz2_Data\\train.csv')/(1024**2):.2f}MB CSV time: ", (csv_end - csv_start))

    parquet_start = timeit.default_timer()
    read_parquet()
    parquet_end = timeit.default_timer()
    print(f"Reading {path.getsize('D:\\Pz2_Data\\train.parquet')/(1024**2):.2f}MB Parquet time: ", (parquet_end - parquet_start))

if __name__ == "__main__":
    #convert_to_parquet()
    main()