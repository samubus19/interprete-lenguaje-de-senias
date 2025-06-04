import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from model.sign_recognizer import SignRecognizer

def plot_training_history(history):
    """Grafica el historial de entrenamiento"""
    plt.figure(figsize=(12, 4))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def load_training_data():
    """Carga los datos de entrenamiento desde los archivos recolectados"""
    data_dir = "data/collected"
    if not os.path.exists(data_dir):
        raise FileNotFoundError("No se encontró el directorio de datos recolectados. Ejecuta collect_data.py primero.")
    
    X = []
    y = []
    class_names = []
    
    # Cargar el archivo de señas
    with open("data/signs.json", "r", encoding="utf-8") as f:
        signs_data = json.load(f)
        class_names = [sign["name"] for sign in signs_data["signs"]]
    
    # Cargar datos de cada directorio de ID
    for id_dir in os.listdir(data_dir):
        id_path = os.path.join(data_dir, id_dir)
        if not os.path.isdir(id_path):
            continue
            
        # Cargar cada archivo de datos
        for filename in os.listdir(id_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(id_path, filename)
                try:
                    landmarks = np.load(file_path)
                    X.append(landmarks)
                    # Crear etiqueta one-hot (asumiendo que el ID corresponde al índice de la seña)
                    label = np.zeros(len(signs_data["signs"]))
                    label[int(id_dir) - 1] = 1  # Restamos 1 porque los IDs empiezan en 1
                    y.append(label)
                except Exception as e:
                    print(f"Error al cargar {filename}: {e}")
    
    return np.array(X), np.array(y), class_names

def main():
    print("Cargando datos de entrenamiento...")
    try:
        X, y, class_names = load_training_data()
        
        if len(X) == 0:
            print("Error: No se encontraron datos de entrenamiento.")
            print("Por favor, ejecuta collect_data.py primero para recolectar datos.")
            return
            
        print(f"\nDatos cargados exitosamente:")
        print(f"- Número de muestras: {len(X)}")
        print(f"- Número de clases: {len(class_names)}")
        print(f"- Clases: {', '.join(class_names)}")
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Inicializar y entrenar el modelo
        print("\nIniciando entrenamiento del modelo...")
        recognizer = SignRecognizer()
        
        history = recognizer.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluar el modelo
        print("\nEvaluando el modelo...")
        test_loss, test_accuracy = recognizer.model.evaluate(X_test, y_test)
        print(f"\nPrecisión en conjunto de prueba: {test_accuracy:.4f}")
        
        # Obtener predicciones
        y_pred = recognizer.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Generar reporte de clasificación
        print("\nReporte de Clasificación:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
        
        # Graficar resultados
        plot_training_history(history)
        plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
        
        # Guardar el modelo
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'sign_language_model.h5')
        recognizer.save_model(model_path)
        print(f"\nModelo guardado en: {model_path}")
        
        # Guardar estadísticas de entrenamiento
        stats = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'num_samples': len(X),
            'num_classes': len(class_names),
            'class_names': class_names
        }
        
        with open('models/training_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        print("\nEntrenamiento completado. Se han generado:")
        print("- Gráfico de historial de entrenamiento (training_history.png)")
        print("- Matriz de confusión (confusion_matrix.png)")
        print("- Estadísticas de entrenamiento (models/training_stats.json)")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main() 