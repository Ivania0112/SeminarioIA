import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import plotly.express as px

# Función para realizar el proceso de clasificación y retornar la precisión
def clasificar(X, y, porcentaje_entrenamiento=0.8):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=1-porcentaje_entrenamiento)
    
    perceptron = Perceptron()
    perceptron.fit(X_entrenamiento, y_entrenamiento)
    
    y_pred = perceptron.predict(X_prueba)
    
    precision = accuracy_score(y_prueba, y_pred)
    return precision

#CARGAR DATOS CVS
#data = pd.read_csv('spheres2d10.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
#data = pd.read_csv('spheres2d50.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
data = pd.read_csv('spheres2d70.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
# Número de particiones
num_particiones = 10

# Lista para almacenar las precisiones
precisiones = []

# Lista para almacenar los datos de las particiones
datos_particiones = []

# Realizar el proceso de particionamiento y clasificación
for _ in range(num_particiones):
    porcentaje_entrenamiento = 0.8  # Porcentaje de entrenamiento (puedes modificarlo si es necesario)
    precision = clasificar(data[['feature1', 'feature2', 'feature3']], data['label'], porcentaje_entrenamiento)
    precisiones.append(precision)
    datos_particiones.append(data[['feature1', 'feature2', 'feature3']].sample(frac=1-porcentaje_entrenamiento))

# Calcular la precisión promedio de las particiones
precision_promedio = sum(precisiones) / num_particiones

# Crear un DataFrame con los datos de las particiones
df_particiones = pd.concat(datos_particiones)

# Agregar la columna 'label' con las etiquetas reales
df_particiones['label'] = data['label']

# Crear una figura con Plotly para mostrar los puntos en 3D
fig = px.scatter_3d(df_particiones, x='feature1', y='feature2', z='feature3', color='label', opacity=0.8,
                     title=f'Clasificación con Precisión Promedio: {precision_promedio:.2f}')
fig.show()