import pandas as pd
import timeit


def read_data() -> pd.DataFrame:
    return pd.read_csv('D:\\Pz3_data\\powerconsumption.csv')


def filtrate_data(data: pd.DataFrame):
    median: pd.DataFrame = pd.DataFrame()

    median = data[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]].median()
    print("Medians")
    print(median, "\n")
    query: str = f"`PowerConsumption_Zone1` > {median["PowerConsumption_Zone1"]} and `PowerConsumption_Zone2` > {median["PowerConsumption_Zone2"]} and `PowerConsumption_Zone3` > {median["PowerConsumption_Zone3"]} "

    filtered: pd.DataFrame = data.query(query)

    print("Filtered")
    print(filtered[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]], "\n")
    
    filtered.to_csv("filtered_data.csv")


def main():
    data: pd.DataFrame = read_data()

    start_time: float = timeit.default_timer()
    filtrate_data(data)
    end_time: float = timeit.default_timer()

    print(f"Worktime {end_time - start_time}")


if __name__ == "__main__":
    main()
