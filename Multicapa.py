import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Crear el modelo de la red neuronal multicapa
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Lectura de los patrones de entrenamiento desde un archivo CSV
data = np.genfromtxt('XOR_trn.csv', delimiter=',')
X_train = data[:, :-1]
y_train = data[:, -1]

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=250)

# Prueba del modelo en datos de prueba
test_data = np.genfromtxt('XOR_tst.csv', delimiter=',')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Graficar los patrones y la región que separa las clases
xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
input_data = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(input_data)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.6)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Clase 0', marker='o')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Clase 1', marker='x')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
