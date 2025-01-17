# Tugas
Tugas AAS R703

import numpy as np

# Fungsi aktivasi (Sigmoid dan turunannya)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Fungsi aktivasi tambahan (ReLU dan turunannya)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Neural Network Sederhana
def train_neural_network(X, y, hidden_neurons, epochs, learning_rate, activation="sigmoid"):
    np.random.seed(1)

    # Pilih fungsi aktivasi
    if activation == "sigmoid":
        activation_function = sigmoid
        activation_derivative = sigmoid_derivative
    elif activation == "relu":
        activation_function = relu
        activation_derivative = relu_derivative
    else:
        raise ValueError("Fungsi aktivasi tidak dikenali. Gunakan 'sigmoid' atau 'relu'.")

    # Inisialisasi bobot secara acak
    input_neurons = X.shape[1]
    output_neurons = y.shape[1]

    weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))

    history = []  # Untuk menyimpan error tiap epoch

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)  # Output layer tetap sigmoid

        # Hitung error
        error = y - output_layer_output

        # Backpropagation
        output_delta = error * sigmoid_derivative(output_layer_output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * activation_derivative(hidden_layer_output)

        # Perbarui bobot
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

        # Simpan error ke dalam history
        history.append(np.mean(np.abs(error)))

        # Tampilkan error setiap 100 epoch
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {np.mean(np.abs(error))}")

    return weights_input_hidden, weights_hidden_output, history

# Prediksi
def predict(X, weights_input_hidden, weights_hidden_output, activation="sigmoid"):
    if activation == "sigmoid":
        activation_function = sigmoid
    elif activation == "relu":
        activation_function = relu
    else:
        raise ValueError("Fungsi aktivasi tidak dikenali. Gunakan 'sigmoid' atau 'relu'.")

    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = activation_function(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)  # Output layer tetap sigmoid

    return output_layer_output

# Program utama
if __name__ == "__main__":
    # Studi Kasus: Prediksi Risiko Kemacetan Berdasarkan Waktu dan Volume Kendaraan
    # Data: [Waktu (jam), Volume Kendaraan] -> [Kemacetan (1) atau Tidak Macet (0)]
    X = np.array([
        [6, 20],  # Pagi, sedikit kendaraan
        [7, 50],  # Pagi, lebih banyak kendaraan
        [8, 80],  # Jam sibuk pagi
        [12, 30], # Siang, volume sedang
        [17, 100], # Jam sibuk sore
        [20, 40]  # Malam, volume rendah
    ])
    y = np.array([
        [0],  # Tidak macet
        [0],  # Tidak macet
        [1],  # Macet
        [0],  # Tidak macet
        [1],  # Macet
        [0]   # Tidak macet
    ])

    # Parameter
    hidden_neurons = 4
    epochs = 10000
    learning_rate = 0.1
    activation_function = "relu"  # Pilih fungsi aktivasi: 'sigmoid' atau 'relu'

    # Latih jaringan
    weights_input_hidden, weights_hidden_output, history = train_neural_network(
        X, y, hidden_neurons, epochs, learning_rate, activation=activation_function
    )

    # Prediksi
    predictions = predict(X, weights_input_hidden, weights_hidden_output, activation=activation_function)

    # Tampilkan hasil pelatihan
    print("\nPrediksi setelah pelatihan:")
    for i, prediction in enumerate(predictions):
        print(f"Input: {X[i]}, Output: {prediction.round(2)}, Target: {y[i]}")

    # Plot error selama pelatihan
    try:
        import matplotlib.pyplot as plt

        plt.plot(range(epochs), history)
        plt.title("Perubahan Error Selama Pelatihan")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
    except ImportError:
        print("Matplotlib tidak tersedia. Instal matplotlib untuk melihat grafik error.")

    # Penyelesaian akhir
    print("\nPenyelesaian:")
    for i, prediction in enumerate(predictions):
        status = "Macet" if prediction >= 0.5 else "Tidak Macet"
        print(f"Input: Waktu = {X[i][0]} Jam, Volume Kendaraan = {X[i][1]} -> Prediksi: {status} (Probabilitas: {prediction[0]:.2f})")
