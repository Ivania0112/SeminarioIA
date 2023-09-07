import numpy as np
import matplotlib.pyplot as plt

# Función de activación (función escalón)
def step_function(x):
    return 1 if x > 0 else 0

# Entrenamiento del perceptrón simple
def train_perceptron(X, y, learning_rate, epochs):
    num_samples, num_features = X.shape
    weights = np.random.rand(num_features)
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(num_samples):
            prediction = step_function(np.dot(X[i], weights))
            error = y[i] - prediction
            total_error += abs(error)
            weights += learning_rate * error * X[i]
        errors.append(total_error)

        if total_error == 0:
            break

    return weights, errors

# Lectura de los patrones de entrenamiento desde un archivo CSV
data = np.genfromtxt('XOR_trn.csv', delimiter=',')
X_train = data[:, :-1]
y_train = data[:, -1]

# Entrenamiento del perceptrón
learning_rate = 0.1
epochs = 1000
trained_weights, training_errors = train_perceptron(X_train, y_train, learning_rate, epochs)

# Prueba del perceptrón en datos de prueba
test_data = np.genfromtxt('XOR_tst.csv', delimiter=',')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

correct_predictions = 0
for i in range(X_test.shape[0]):
    prediction = step_function(np.dot(X_test[i], trained_weights))
    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print("Accuracy:", accuracy)

# Graficar los patrones y la recta que los separa
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Clase 0', marker='o')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Clase 1', marker='x')

# Dibujar la recta que separa los patrones
if trained_weights[0] != 0:
    x_line = np.linspace(-1, 1, 100)
    y_line = (-trained_weights[0] * x_line) / trained_weights[1]
    plt.plot(x_line, y_line, '-r', label='Recta separadora')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
