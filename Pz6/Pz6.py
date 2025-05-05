import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns

csv_files = [
    ["Data/Pz5_Background.csv", "Background"],
    ["Data/Pz5_Fridge.csv", "Fridge"],
    ["Data/Pz5_Laptop.csv", "Laptop"],
    ["Data/Pz5_Charger_45W.csv", "Charger"],
    ["Data/Pz5_Microwave_on.csv", "Microwave"],
    ["Data/Pz5_Wireless_charger.csv", "Wireless charger"],
    ["Data/Pz5_WI-FI_router.csv", "WI-FI router"]
]

df_list = []

for file, name in csv_files:
    df = pd.read_csv(file)
    df["Device"] = name
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)



def search_anomaly():
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='ms')

    X = combined_df[['MagneticRotationSensor']]
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    combined_df['IF_Anomaly'] = iso_forest.fit_predict(X)
    combined_df['IF_Anomaly'] = combined_df['IF_Anomaly'] == -1

    plt.figure(figsize=(16, 6))
    sns.lineplot(x='timestamp', y='MagneticRotationSensor', hue='Device', data=combined_df, legend=True)
    plt.scatter(
        combined_df[combined_df['IF_Anomaly']]['timestamp'],
        combined_df[combined_df['IF_Anomaly']]['MagneticRotationSensor'],
        color='red', label='Isolation Forest Anomalies', s=30
    )
    plt.title('Виявлення аномалій за допомогою Isolation Forest')
    plt.xlabel('Час')
    plt.ylabel('Магнітне поле (μT)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_room():
    device_means = combined_df.groupby('Device')['MagneticRotationSensor'].mean()

    device_positions = {
        "Background": (1, 1),
        "Fridge": (3, 1),
        "Laptop": (1, 4),
        "Charger": (3, 3),
        "Microwave": (3, 2),
        "Wireless charger": (2,4),
        "WI-FI router": (3,4)
    }

    x, y, values = [], [], []

    for device, (xi, yi) in device_positions.items():
        if device in device_means:
            x.append(xi)
            y.append(yi)
            values.append(device_means[device])

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x, y, c=values, s=1000, cmap='coolwarm', edgecolors='black')
    for i, device in enumerate(device_positions.keys()):
        plt.text(x[i], y[i], device, ha='center', va='center', color='black', fontsize=10, weight='bold')

    plt.colorbar(sc, label='Магнітне поле (μT)')
    plt.title('Умовна карта розподілу магнітного поля в кімнаті')
    plt.xlim(0, 4)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal')
    plt.show()

search_anomaly()
draw_room()