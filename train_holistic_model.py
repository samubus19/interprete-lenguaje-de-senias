import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

class HolisticSignRecognizer:
    def __init__(self, num_classes):
        """
        Reconocedor de señas usando características holísticas (378 features)
        
        Args:
            num_classes (int): Número de clases/frases
        """
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Construye el modelo de reconocimiento usando características holísticas"""
        input_shape = (378,)  # 132 (pose) + 63 (mano izq) + 63 (mano der) + 120 (cara)
        
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model

def plot_training_history(history):
    """Grafica el historial de entrenamiento"""
    plt.figure(figsize=(15, 5))
    
    # Gráfico de precisión
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdida
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de Top-K accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history.history['top_k_categorical_accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_top_k_categorical_accuracy'], label='Validación')
    plt.title('Top-K Precisión')
    plt.xlabel('Época')
    plt.ylabel('Top-K Precisión')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_holistic.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión - Modelo Holístico')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_holistic.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_holistic_training_data(landmarks_dir="landmarks_data"):
    """
    Carga los datos de entrenamiento desde los landmarks extraídos
    
    Args:
        landmarks_dir (str): Directorio con datos de landmarks
        
    Returns:
        tuple: (X, y, class_names, phrase_to_id)
    """
    if not os.path.exists(landmarks_dir):
        raise FileNotFoundError(f"No se encontró el directorio {landmarks_dir}. Ejecuta landmark_extractor.py primero.")
    
    X = []
    y = []
    class_names = []
    phrase_to_id = {}
    
    # Obtener todas las frases (directorios)
    phrase_dirs = [d for d in os.listdir(landmarks_dir) 
                  if os.path.isdir(os.path.join(landmarks_dir, d))]
    
    print(f"Frases encontradas: {phrase_dirs}")
    
    # Asignar IDs a las frases
    for phrase_id, phrase_name in enumerate(sorted(phrase_dirs)):
        class_names.append(phrase_name)
        phrase_to_id[phrase_name] = phrase_id
        
        phrase_path = os.path.join(landmarks_dir, phrase_name)
        
        # Cargar archivos de landmarks de esta frase
        landmark_files = [f for f in os.listdir(phrase_path) if f.endswith('_landmarks.pkl')]
        
        print(f"Procesando frase '{phrase_name}': {len(landmark_files)} videos")
        
        for landmark_file in landmark_files:
            landmark_path = os.path.join(phrase_path, landmark_file)
            
            try:
                with open(landmark_path, 'rb') as f:
                    sequence_data = pickle.load(f)
                
                # Extraer vectores de características de cada frame válido
                feature_vectors = sequence_data['feature_vectors']
                
                # Agregar cada frame como una muestra individual
                for feature_vector in feature_vectors:
                    # Solo agregar frames con detecciones válidas (no todos ceros)
                    if np.any(feature_vector):
                        X.append(feature_vector)
                        y.append(phrase_id)
                        
            except Exception as e:
                print(f"Error al cargar {landmark_file}: {e}")
    
    if len(X) == 0:
        raise ValueError("No se encontraron datos válidos. Verifica que landmark_extractor.py haya procesado correctamente los datos.")
    
    return np.array(X), np.array(y), class_names, phrase_to_id

def main():
    print("=== ENTRENAMIENTO DE MODELO HOLÍSTICO LSA ===")
    print("Cargando datos de entrenamiento...")
    
    try:
        X, y, class_names, phrase_to_id = load_holistic_training_data()
        
        print(f"\nDatos cargados exitosamente:")
        print(f"- Número de muestras: {len(X)}")
        print(f"- Número de clases: {len(class_names)}")
        print(f"- Dimensión de características: {X.shape[1]}")
        print(f"- Clases: {', '.join(class_names)}")
        
        # Verificar distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nDistribución de muestras por clase:")
        for class_id, count in zip(unique, counts):
            print(f"- {class_names[class_id]}: {count} muestras")
        
        # Convertir labels a one-hot
        y_categorical = to_categorical(y, num_classes=len(class_names))
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nDivisión de datos:")
        print(f"- Entrenamiento: {len(X_train)} muestras")
        print(f"- Prueba: {len(X_test)} muestras")
        
        # Inicializar y entrenar el modelo
        print("\nIniciando entrenamiento del modelo...")
        recognizer = HolisticSignRecognizer(num_classes=len(class_names))
        
        print("\nArquitectura del modelo:")
        recognizer.model.summary()
        
        # Callbacks para entrenamiento
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        history = recognizer.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar el modelo
        print("\n=== EVALUANDO MODELO ===")
        test_loss, test_accuracy, test_top_k = recognizer.model.evaluate(X_test, y_test, verbose=0)
        print(f"Pérdida en conjunto de prueba: {test_loss:.4f}")
        print(f"Precisión en conjunto de prueba: {test_accuracy:.4f}")
        print(f"Top-K Precisión: {test_top_k:.4f}")
        
        # Obtener predicciones
        y_pred = recognizer.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Generar reporte de clasificación
        print(f"\n=== REPORTE DE CLASIFICACIÓN ===")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_names, zero_division=0))
        
        # Graficar resultados
        print(f"\n=== GENERANDO GRÁFICOS ===")
        plot_training_history(history)
        plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
        
        # Guardar el modelo
        print(f"\n=== GUARDANDO MODELO ===")
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'holistic_sign_model.h5')
        recognizer.model.save(model_path)
        
        # Guardar metadatos
        metadata = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'test_top_k_accuracy': float(test_top_k),
            'num_samples': len(X),
            'num_classes': len(class_names),
            'class_names': class_names,
            'phrase_to_id': phrase_to_id,
            'feature_dim': 378,
            'model_type': 'holistic_dense'
        }
        
        metadata_path = os.path.join('models', 'holistic_model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        print(f"\n=== RESUMEN FINAL ===")
        print(f"Modelo entrenado: Red densa holística")
        print(f"Precisión final: {test_accuracy:.4f}")
        print(f"Top-K Precisión: {test_top_k:.4f}")
        print(f"Número de clases: {len(class_names)}")
        print(f"Características por muestra: 378")
        
        print(f"\nArchivos generados:")
        print(f"- Modelo: {model_path}")
        print(f"- Metadatos: {metadata_path}")
        print(f"- Historial: training_history_holistic.png")
        print(f"- Matriz de confusión: confusion_matrix_holistic.png")
        
        print(f"\n¡Entrenamiento completado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
