import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

class LSALSTMTrainer:
    def __init__(self, max_sequence_length=30, keypoints_length=1662):
        """
        Entrenador de modelo LSTM para LSA
        
        Args:
            max_sequence_length (int): Longitud máxima de secuencia
            keypoints_length (int): Número de keypoints por frame
        """
        self.max_sequence_length = max_sequence_length
        self.keypoints_length = keypoints_length
        self.model = None
        self.class_names = []
        self.history = None
    
    def build_lstm_model(self, num_classes):
        """
        Construye el modelo LSTM basado en el repositorio de referencia
        
        Args:
            num_classes (int): Número de clases
            
        Returns:
            tensorflow.keras.Model: Modelo LSTM compilado
        """
        model = Sequential()
        
        # Primera capa LSTM
        model.add(LSTM(64, 
                      return_sequences=True, 
                      input_shape=(self.max_sequence_length, self.keypoints_length),
                      kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        
        # Segunda capa LSTM
        model.add(LSTM(128, 
                      return_sequences=False,
                      kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        
        # Capas densas
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        
        # Capa de salida
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compilar modelo
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_dataset(self, dataset_path):
        """
        Carga el dataset procesado
        
        Args:
            dataset_path (str): Ruta al archivo del dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_names)
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset['X']
        y = dataset['y']
        self.class_names = dataset['class_names']
        
        print(f"Dataset cargado:")
        print(f"- Secuencias: {X.shape}")
        print(f"- Clases: {len(self.class_names)}")
        print(f"- Nombres de clases: {self.class_names}")
        
        # Convertir labels a one-hot
        y_categorical = to_categorical(y, num_classes=len(self.class_names))
        
        # División train/test (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"División de datos:")
        print(f"- Entrenamiento: {X_train.shape[0]} secuencias")
        print(f"- Prueba: {X_test.shape[0]} secuencias")
        
        return X_train, X_test, y_train, y_test, self.class_names
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Entrena el modelo LSTM
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            
        Returns:
            History: Historial de entrenamiento
        """
        if self.model is None:
            raise ValueError("El modelo debe ser construido antes del entrenamiento")
        
        print(f"\n=== ENTRENANDO MODELO LSTM ===")
        print(f"Épocas: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_lstm_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo entrenado
        
        Args:
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de la evaluación")
        
        print(f"\n=== EVALUANDO MODELO ===")
        
        # Predicciones
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Métricas
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Pérdida de prueba: {test_loss:.4f}")
        print(f"Precisión de prueba: {test_accuracy:.4f}")
        
        # Reporte de clasificación
        print(f"\n=== REPORTE DE CLASIFICACIÓN ===")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.class_names, zero_division=0))
        
        # Matriz de confusión
        self._plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Grafica la matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Matriz de Confusión - Modelo LSTM LSA')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_lstm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Matriz de confusión guardada como: confusion_matrix_lstm.png")
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Precisión
        axes[0].plot(self.history.history['accuracy'], label='Entrenamiento')
        axes[0].plot(self.history.history['val_accuracy'], label='Validación')
        axes[0].set_title('Precisión del Modelo')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Precisión')
        axes[0].legend()
        axes[0].grid(True)
        
        # Pérdida
        axes[1].plot(self.history.history['loss'], label='Entrenamiento')
        axes[1].plot(self.history.history['val_loss'], label='Validación')
        axes[1].set_title('Pérdida del Modelo')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Pérdida')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_lstm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Historial de entrenamiento guardado como: training_history_lstm.png")
    
    def save_model(self, model_path, metadata_path=None):
        """
        Guarda el modelo y metadatos
        
        Args:
            model_path (str): Ruta para guardar el modelo
            metadata_path (str): Ruta para guardar metadatos
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Guardar modelo
        self.model.save(model_path)
        
        # Guardar metadatos
        if metadata_path is None:
            metadata_path = model_path.replace('.h5', '_metadata.json')
        
        metadata = {
            'max_sequence_length': self.max_sequence_length,
            'keypoints_length': self.keypoints_length,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_type': 'lstm'
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        print(f"Modelo guardado en: {model_path}")
        print(f"Metadatos guardados en: {metadata_path}")

def main():
    """Función principal de entrenamiento"""
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar modelo LSTM para LSA')
    parser.add_argument('--dataset', type=str, choices=['original', 'augmented', 'auto'],
                       default='auto', help='Tipo de dataset a usar')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Saltar data augmentation automático')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    
    args = parser.parse_args()
    
    print("=== ENTRENAMIENTO DE MODELO LSTM PARA LSA ===")
    
    # Verificar datasets disponibles
    original_dataset = "models/sequence_dataset.pkl"
    augmented_dataset = "models/sequence_dataset_augmented.pkl"
    
    # Determinar qué dataset usar
    if args.dataset == 'original':
        if os.path.exists(original_dataset):
            dataset_path = original_dataset
            print(f"📝 Usando dataset original (forzado): {dataset_path}")
        else:
            print(f"❌ Dataset original no encontrado: {original_dataset}")
            return
            
    elif args.dataset == 'augmented':
        if os.path.exists(augmented_dataset):
            dataset_path = augmented_dataset
            print(f"🚀 Usando dataset aumentado (forzado): {dataset_path}")
        else:
            print(f"❌ Dataset aumentado no encontrado: {augmented_dataset}")
            print("Ejecuta: python data_augmentation.py")
            return
            
    else:  # auto
        if os.path.exists(augmented_dataset):
            dataset_path = augmented_dataset
            print(f"✅ Dataset aumentado encontrado: {dataset_path}")
            
            # Preguntar si quiere usar el original en su lugar (solo en modo interactivo)
            if not args.no_augmentation:
                use_original = input("¿Quieres usar el dataset original sin aumentar? (y/N): ").lower().strip()
                if use_original == 'y' or use_original == 'yes':
                    if os.path.exists(original_dataset):
                        dataset_path = original_dataset
                        print(f"📝 Usando dataset original: {dataset_path}")
                    else:
                        print(f"❌ Dataset original no encontrado: {original_dataset}")
                        return
                else:
                    print(f"🚀 Usando dataset aumentado: {dataset_path}")
            else:
                print(f"🚀 Usando dataset aumentado (automático): {dataset_path}")
                
        elif os.path.exists(original_dataset):
            dataset_path = original_dataset
            print(f"⚠️ Solo dataset original encontrado: {dataset_path}")
            
            # Sugerir data augmentation (solo si no está deshabilitado)
            if not args.no_augmentation:
                print("💡 Recomendación: Ejecuta 'python data_augmentation.py' para más datos")
                use_augmentation = input("¿Quieres ejecutar data augmentation ahora? (Y/n): ").lower().strip()
                
                if use_augmentation != 'n' and use_augmentation != 'no':
                    print("🔄 Ejecutando data augmentation...")
                    try:
                        import subprocess
                        result = subprocess.run(['python', 'data_augmentation.py'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print("✅ Data augmentation completado")
                            dataset_path = augmented_dataset
                        else:
                            print(f"❌ Error en data augmentation: {result.stderr}")
                            print("Continuando con dataset original...")
                    except Exception as e:
                        print(f"❌ Error ejecutando data augmentation: {e}")
                        print("Continuando con dataset original...")
            
        else:
            print(f"❌ No se encontró ningún dataset")
            print("Ejecuta primero: python sequence_data_processor.py")
            print("Luego (opcional): python data_augmentation.py")
            return
    
    try:
        # Crear entrenador
        trainer = LSALSTMTrainer(max_sequence_length=30, keypoints_length=1662)
        
        # Cargar datos
        X_train, X_test, y_train, y_test, class_names = trainer.load_dataset(dataset_path)
        
        # Construir modelo
        model = trainer.build_lstm_model(num_classes=len(class_names))
        
        print(f"\n=== ARQUITECTURA DEL MODELO ===")
        model.summary()
        
        # Entrenar modelo
        history = trainer.train_model(X_train, y_train, X_test, y_test, 
                                    epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluar modelo
        results = trainer.evaluate_model(X_test, y_test)
        
        # Generar gráficos
        trainer.plot_training_history()
        
        # Guardar modelo
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/lsa_lstm_model.h5')
        
        # Resumen final
        print(f"\n=== RESUMEN FINAL ===")
        print(f"Modelo: LSTM para LSA")
        print(f"Precisión final: {results['test_accuracy']:.4f}")
        print(f"Número de clases: {len(class_names)}")
        print(f"Clases: {', '.join(class_names)}")
        
        print(f"\nArchivos generados:")
        print(f"- Modelo: models/lsa_lstm_model.h5")
        print(f"- Metadatos: models/lsa_lstm_model_metadata.json")
        print(f"- Mejor modelo: models/best_lstm_model.h5")
        print(f"- Historial: training_history_lstm.png")
        print(f"- Matriz de confusión: confusion_matrix_lstm.png")
        
        print(f"\n✅ ¡Entrenamiento completado exitosamente!")
        print(f"Siguiente paso: python app_lstm.py")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
