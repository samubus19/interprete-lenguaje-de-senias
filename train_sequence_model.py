import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

N_FRAMES = 20  # Debe coincidir con el recolector

# Cargar señas
with open("data/signs.json", "r", encoding="utf-8") as f:
    signs_data = json.load(f)
    class_names = [sign["name"] for sign in signs_data["signs"]]
    num_classes = len(class_names)

# Cargar secuencias y etiquetas
def load_sequences(data_dir="data/collected_sequences"):
    X, y = [], []
    for sign_id, sign in enumerate(class_names):
        sign_dir = os.path.join(data_dir, str(sign_id))
        if not os.path.exists(sign_dir):
            continue
        for filename in os.listdir(sign_dir):
            if filename.endswith(".npy"):
                seq = np.load(os.path.join(sign_dir, filename))
                if seq.shape == (N_FRAMES, 42):  # 21 landmarks x 2
                    X.append(seq)
                    y.append(sign_id)
    return np.array(X), np.array(y)

print("Cargando secuencias...")
X, y = load_sequences()
print("X:")
print(X)
print("y:")
print(y)
print(f"Total de secuencias: {len(X)}")

# Mostrar información sobre las clases
unique_classes = np.unique(y)
print(f"Clases únicas encontradas: {unique_classes}")
print(f"Número de clases: {len(unique_classes)}")

for class_id in unique_classes:
    count = np.sum(y == class_id)
    print(f"Clase {class_id} ({class_names[class_id]}): {count} secuencias")

if len(X) == 0:
    print("No hay datos de secuencias. Ejecuta sequence_data_collector.py primero.")
    exit(1)

# One-hot encoding de etiquetas
y_cat = to_categorical(y, num_classes=num_classes)
print("y_cat:")
print(y_cat)

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)

# Modelo LSTM mejorado
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(N_FRAMES, 42)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitectura del modelo:")
model.summary()

print("\nEntrenando modelo LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,  # Reducido para mejor generalización
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

# Evaluación
print("\nEvaluando el modelo...")
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión en test: {acc:.4f}")

# Reporte de clasificación
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Asegurarse de que todas las clases estén presentes
unique_pred_classes = np.unique(y_pred_classes)
unique_true_classes = np.unique(y_true_classes)
print(f"\nClases predichas: {unique_pred_classes}")
print(f"Clases verdaderas: {unique_true_classes}")

print("\nReporte de Clasificación:")
# print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

# Matriz de confusión
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('confusion_matrix_seq.png')
    plt.close()

plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

# Guardar el modelo
os.makedirs('models', exist_ok=True)
model.save('models/sign_language_sequence_model.h5')
print("\nModelo LSTM de secuencias guardado en models/sign_language_sequence_model.h5") 