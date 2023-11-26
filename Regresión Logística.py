import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# INICIALIZACION Y ENTRENAMIENTO
logreg = LogisticRegression(max_iter=10000)  # Ajusta el número máximo de iteraciones si es necesario
logreg.fit(X_train, y_train)

# PREDICCIONES
y_pred = logreg.predict(X_test)

# METRICAS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# MATRIZ DE CONFUSION
conf_matrix = confusion_matrix(y_test, y_pred)

# METRICAS ADICIONALES
true_negatives = conf_matrix[0, 0]
false_negatives = conf_matrix[1, 0]
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

# IMPRESION DE METRICAS
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
