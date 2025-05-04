import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d, UnivariateSpline
from keras import models, layers, losses
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


np.random.seed(42)
time = np.arange(0, 100, 0.5)
consumption = np.sin(time / 5) + np.random.normal(0, 0.1, len(time))

missing_mask = np.random.rand(len(consumption)) < 0.2
observed = consumption.copy()
observed[missing_mask] = np.nan

missing_indices = np.where(missing_mask)[0]
observed_indices = np.where(~missing_mask)[0]

scaler = MinMaxScaler()
consumption_scaled = scaler.fit_transform(consumption.reshape(-1, 1)).flatten()
observed_scaled = scaler.transform(observed.reshape(-1, 1)).flatten()

input_data = observed_scaled.copy()
input_data[np.isnan(input_data)] = 0
scaled_down = input_data[observed_indices]

noisy_input = scaled_down + np.random.normal(0, 0.1, size=scaled_down.shape)


def interpolate(method):
    x_known = time[observed_indices]
    y_known = observed[observed_indices]

    if method == 'linear':
        f = interp1d(x_known, y_known, kind='linear', fill_value='extrapolate')
        return f(time)
    elif method == 'poly':
        poly = np.poly1d(np.polyfit(x_known, y_known, deg=10))
        return poly(time)
    elif method == 'spline':
        spline = UnivariateSpline(x_known, y_known, s=1)
        return spline(time)
    else:
        raise ValueError("Unsupported method")

def create_autoencoder():
    hidden_size = 100
    middle_size = 32

    input_layer = layers.Input(shape=(1,))
    hidden = layers.Dense(hidden_size, activation='relu')(input_layer)
    middle = layers.Dense(middle_size, activation='relu')(hidden)
    encoder = models.Model(inputs=input_layer, outputs=middle, name='encoder')

    decoder_input = layers.Input(shape=(middle_size,))
    upsampled = layers.Dense(hidden_size, activation='relu')(decoder_input)
    output = layers.Dense(1, activation='relu')(upsampled)
    decoder = models.Model(inputs=decoder_input, outputs=output, name='decoder')

    autoencoder = models.Model(inputs=encoder.input, outputs=decoder(encoder.output))
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.summary()
    return autoencoder

def evaluate(name, reconstructed_series):
    return np.sqrt(mean_squared_error(consumption[missing_indices], reconstructed_series[missing_indices]))

autoencoder = create_autoencoder()
autoencoder.fit(noisy_input, scaled_down, epochs=100, batch_size=32, verbose=0)

interp_linear = interpolate('linear')
interp_poly = interpolate('poly')
interp_spline = interpolate('spline')

scaled = scaler.fit_transform(interp_linear.reshape(-1, 1)).flatten()
reconstructed_scaled = autoencoder.predict(scaled.reshape(-1, 1)).flatten()
reconstructed = scaler.inverse_transform(reconstructed_scaled.reshape(-1, 1)).flatten()

results = {
    'Autoencoder': evaluate('Autoencoder', reconstructed),
    'Polynomial': evaluate('Polynomial', interp_poly),
    'Spline': evaluate('Spline', interp_spline),
    'Linear': evaluate('Linear', interp_linear)
}

def draw():
    plt.figure(figsize=(14, 6))
    plt.plot(time, consumption, label='Original', alpha=0.5)
    plt.plot(time, observed, 'o', label='Observed (with missing)', markersize=3)
    plt.plot(time, reconstructed, label='Autoencoder')
    plt.plot(time, interp_spline, '--', label='Spline')
    plt.plot(time, interp_poly, 'x', label='Polynomial')
    plt.plot(time, interp_linear, label='Linear')
    plt.title('Energy Consumption Imputation')
    plt.legend()
    plt.show()


print("RMSE on missing values:")
for method, score in results.items():
    print(f"{method}: {score:.4f}")

draw()
