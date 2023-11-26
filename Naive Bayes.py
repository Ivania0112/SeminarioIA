import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

# INICIALIZACION Y ENTRENAMIENTO
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# PREDICCIONES
y_pred = naive_bayes.predict(X_test)

# METRICAS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# MATRIZ DE CONFUSION
conf_matrix = confusion_matrix(y_test, y_pred)

# IMPRESION DE METRICAS
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
