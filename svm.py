import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados de exemplo do Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Considerar apenas duas classes para um problema binário (SVM binário)
X = X[y != 2]
y = y[y != 2]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir o modelo SVM com um kernel linear
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01))  # L2 regularização para o efeito do SVM
])

# Compilar o modelo (usando hinge loss para SVM)
model.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, verbose=2)

# Avaliar no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
