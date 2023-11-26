import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# CARGA DEL CONJUNTO DE DATOS
data = pd.read_csv('zoo.data', header=None)

# ASIGNACION DE COLUMNAS AL DATASET
columns = ['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
           'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type']
data.columns = columns

# ETIQUETAS Y CARACTERISTICAS (X) ETIQUETAS (Y)
X = data.drop(['animal_name', 'class_type'], axis=1)
y = data['class_type']

# CONJUNTOS ENTRENAMIENTO Y PRUEBAS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# NORMALIZACION DE DATOS
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# ONE-HOT ENCODING
num_classes = len(np.unique(y))
y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=num_classes)

# ESTRUCTURA RED NEURONAL
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# COMPILACION DE MODELO
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ENTRENAMIENTO RED NEURONAL
batch_size = 32
epochs = 100
model.fit(X_train_normalized, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test_normalized, y_test))

# EVALUACION DEL MODELO
y_pred_probs = model.predict(X_test_normalized)
y_pred = np.argmax(y_pred_probs, axis=1) + 1

# CALCULO DE METRICAS
accuracy = accuracy_score(np.argmax(y_test, axis=1) + 1, y_pred)
precision = precision_score(np.argmax(y_test, axis=1) + 1, y_pred, average='weighted', zero_division=0)
recall = recall_score(np.argmax(y_test, axis=1) + 1, y_pred, average='weighted', zero_division=0)
f1 = f1_score(np.argmax(y_test, axis=1) + 1, y_pred, average='weighted')

# MATRIZ DE CONFUSION
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1) + 1, y_pred)

# IMPRESION DE METRICAS
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
