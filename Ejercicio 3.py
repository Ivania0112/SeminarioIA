import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Cargar el conjunto de datos
data = pd.read_csv('concentlite.csv', header=None)
data.columns = ['feature1', 'feature2', 'clase']

# Dividir los datos en características (X) y etiquetas (y)
X = data[['feature1', 'feature2']].values
y = data['clase'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir la arquitectura de la red neuronal (por ejemplo, 3 capas con 10 neuronas en cada capa)
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

# Inicializar los pesos y los sesgos de la red
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_input_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_hidden_output = np.zeros((1, output_size))

# Definir la función de activación (por ejemplo, sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definir la tasa de aprendizaje y el número de épocas
learning_rate = 0.01
epochs = 1000

# Entrenar la red neuronal
for epoch in range(epochs):
    # Propagación hacia adelante
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_input_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_hidden_output

    # Calcular la pérdida (error cuadrático medio)
    loss = np.mean((output - y_train.reshape(-1, 1))**2)

    # Retropropagación
    d_output = 2 * (output - y_train.reshape(-1, 1)) / len(X_train)
    d_hidden = np.dot(d_output, weights_hidden_output.T) * (hidden_output * (1 - hidden_output))
    
    # Actualizar los pesos y los sesgos
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, d_output)
    bias_hidden_output -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, d_hidden)
    bias_input_hidden -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

# Realizar predicciones en el conjunto de prueba
hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_input_hidden
hidden_output_test = sigmoid(hidden_input_test)
output_test = np.dot(hidden_output_test, weights_hidden_output) + bias_hidden_output

# Redondear las predicciones a 0 o 1 (clasificación binaria)
predictions = (output_test > 0.5).astype(int)

# Crear un DataFrame de pandas con las predicciones
df_results = pd.DataFrame({'feature1': X_test[:, 0], 'feature2': X_test[:, 1], 'clase': predictions.flatten()})

# Utilizar plotly para visualizar los resultados
fig = px.scatter(df_results, x='feature1', y='feature2', color='clase', title='Clasificación con Perceptrón Multicapa')
fig.show()
