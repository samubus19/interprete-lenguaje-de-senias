import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import pandas as pd
from datetime import datetime

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
        Evalúa el modelo entrenado con análisis completo
        
        Args:
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Métricas de evaluación completas
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de la evaluación")
        
        print(f"\n=== EVALUANDO MODELO ===")
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Métricas básicas
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Pérdida de prueba: {test_loss:.4f}")
        print(f"Precisión de prueba: {test_accuracy:.4f}")
        
        # Análisis detallado por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average=None, zero_division=0
        )
        
        # Métricas promedio
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        print(f"\n=== MÉTRICAS DETALLADAS ===")
        print(f"Precisión macro: {precision_macro:.4f}")
        print(f"Recall macro: {recall_macro:.4f}")
        print(f"F1-Score macro: {f1_macro:.4f}")
        
        # Análisis de confianza
        confidence_analysis = self._analyze_confidence(y_pred_proba, y_true_classes, y_pred_classes)
        
        # Reporte de clasificación
        print(f"\n=== REPORTE DE CLASIFICACIÓN ===")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.class_names, zero_division=0))
        
        # Análisis de overfitting/underfitting
        overfitting_analysis = self._analyze_overfitting()
        
        # Generar todos los gráficos
        self._plot_confusion_matrix(y_true_classes, y_pred_classes)
        self._plot_class_performance(precision, recall, f1, support)
        self._plot_confidence_distribution(y_pred_proba, y_true_classes)
        self._plot_prediction_errors(y_pred_proba, y_true_classes, y_pred_classes)
        
        # Guardar reporte detallado
        detailed_report = self._generate_detailed_report(
            test_loss, test_accuracy, precision, recall, f1, support,
            confidence_analysis, overfitting_analysis, y_true_classes, y_pred_classes, y_pred_proba
        )
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes,
            'y_pred_proba': y_pred_proba,
            'confidence_analysis': confidence_analysis,
            'overfitting_analysis': overfitting_analysis,
            'detailed_report': detailed_report
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
    
    def _analyze_confidence(self, y_pred_proba, y_true, y_pred):
        """Analiza la distribución de confianza de las predicciones"""
        # Confianza de predicciones correctas vs incorrectas
        max_proba = np.max(y_pred_proba, axis=1)
        correct_mask = (y_true == y_pred)
        
        correct_confidence = max_proba[correct_mask]
        incorrect_confidence = max_proba[~correct_mask]
        
        avg_correct_conf = np.mean(correct_confidence) if len(correct_confidence) > 0 else 0
        avg_incorrect_conf = np.mean(incorrect_confidence) if len(incorrect_confidence) > 0 else 0
        
        print(f"\n=== ANÁLISIS DE CONFIANZA ===")
        print(f"Confianza promedio (correctas): {avg_correct_conf:.4f}")
        print(f"Confianza promedio (incorrectas): {avg_incorrect_conf:.4f}")
        print(f"Predicciones con confianza > 0.9: {np.sum(max_proba > 0.9)}/{len(max_proba)} ({np.sum(max_proba > 0.9)/len(max_proba)*100:.1f}%)")
        print(f"Predicciones con confianza < 0.5: {np.sum(max_proba < 0.5)}/{len(max_proba)} ({np.sum(max_proba < 0.5)/len(max_proba)*100:.1f}%)")
        
        return {
            'avg_correct_confidence': avg_correct_conf,
            'avg_incorrect_confidence': avg_incorrect_conf,
            'high_confidence_count': np.sum(max_proba > 0.9),
            'low_confidence_count': np.sum(max_proba < 0.5),
            'confidence_distribution': max_proba
        }
    
    def _analyze_overfitting(self):
        """Analiza si hay overfitting/underfitting basado en el historial"""
        if self.history is None:
            return {'status': 'No hay historial disponible'}
        
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        # Últimas 10 épocas para análisis
        final_epochs = min(10, len(train_acc))
        final_train_acc = np.mean(train_acc[-final_epochs:])
        final_val_acc = np.mean(val_acc[-final_epochs:])
        final_train_loss = np.mean(train_loss[-final_epochs:])
        final_val_loss = np.mean(val_loss[-final_epochs:])
        
        acc_gap = final_train_acc - final_val_acc
        loss_gap = final_val_loss - final_train_loss
        
        # Determinar estado
        if acc_gap > 0.1 and loss_gap > 0.5:
            status = "OVERFITTING SEVERO"
            recommendation = "Reducir complejidad del modelo, más regularización, más datos"
        elif acc_gap > 0.05 and loss_gap > 0.2:
            status = "OVERFITTING MODERADO"
            recommendation = "Aumentar dropout, más datos de entrenamiento"
        elif final_val_acc < 0.6:
            status = "UNDERFITTING"
            recommendation = "Modelo muy simple, aumentar complejidad o más épocas"
        else:
            status = "BUEN BALANCE"
            recommendation = "Modelo bien ajustado"
        
        print(f"\n=== ANÁLISIS DE OVERFITTING ===")
        print(f"Estado: {status}")
        print(f"Gap de precisión (train-val): {acc_gap:.4f}")
        print(f"Gap de pérdida (val-train): {loss_gap:.4f}")
        print(f"Recomendación: {recommendation}")
        
        return {
            'status': status,
            'accuracy_gap': acc_gap,
            'loss_gap': loss_gap,
            'recommendation': recommendation,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc
        }
    
    def _plot_class_performance(self, precision, recall, f1, support):
        """Gráfico de rendimiento por clase"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precisión por clase
        axes[0,0].bar(self.class_names, precision, color='skyblue')
        axes[0,0].set_title('Precisión por Clase')
        axes[0,0].set_ylabel('Precisión')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1)
        
        # Recall por clase
        axes[0,1].bar(self.class_names, recall, color='lightgreen')
        axes[0,1].set_title('Recall por Clase')
        axes[0,1].set_ylabel('Recall')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)
        
        # F1-Score por clase
        axes[1,0].bar(self.class_names, f1, color='orange')
        axes[1,0].set_title('F1-Score por Clase')
        axes[1,0].set_ylabel('F1-Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylim(0, 1)
        
        # Soporte (número de muestras) por clase
        axes[1,1].bar(self.class_names, support, color='coral')
        axes[1,1].set_title('Número de Muestras por Clase')
        axes[1,1].set_ylabel('Cantidad')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('class_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Análisis por clase guardado como: class_performance_analysis.png")
    
    def _plot_confidence_distribution(self, y_pred_proba, y_true):
        """Gráfico de distribución de confianza"""
        max_proba = np.max(y_pred_proba, axis=1)
        
        plt.figure(figsize=(12, 8))
        
        # Histograma general
        plt.subplot(2, 2, 1)
        plt.hist(max_proba, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribución de Confianza General')
        plt.xlabel('Confianza Máxima')
        plt.ylabel('Frecuencia')
        
        # Confianza por clase
        plt.subplot(2, 2, 2)
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_confidence = max_proba[class_mask]
                plt.hist(class_confidence, bins=10, alpha=0.6, label=class_name)
        plt.title('Confianza por Clase')
        plt.xlabel('Confianza')
        plt.ylabel('Frecuencia')
        plt.legend()
        
        # Box plot por clase
        plt.subplot(2, 1, 2)
        confidence_by_class = []
        labels = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                confidence_by_class.append(max_proba[class_mask])
                labels.append(class_name)
        
        plt.boxplot(confidence_by_class, labels=labels)
        plt.title('Distribución de Confianza por Clase (Box Plot)')
        plt.ylabel('Confianza')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Análisis de confianza guardado como: confidence_analysis.png")
    
    def _plot_prediction_errors(self, y_pred_proba, y_true, y_pred):
        """Análisis de errores de predicción"""
        # Encontrar errores
        error_mask = (y_true != y_pred)
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) == 0:
            print("¡No hay errores de predicción!")
            return
        
        # Matriz de errores más detallada
        plt.figure(figsize=(15, 10))
        
        # Heatmap de probabilidades para errores
        plt.subplot(2, 2, 1)
        error_probs = y_pred_proba[error_indices]
        if len(error_probs) > 0:
            sns.heatmap(error_probs[:min(20, len(error_probs))], 
                       xticklabels=self.class_names, 
                       yticklabels=[f"Error {i}" for i in range(min(20, len(error_probs)))],
                       cmap='Reds', annot=True, fmt='.2f')
            plt.title('Probabilidades en Predicciones Erróneas (Top 20)')
        
        # Análisis de confusión más detallado
        plt.subplot(2, 2, 2)
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusión (%)')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        
        # Errores por clase
        plt.subplot(2, 1, 2)
        error_counts = []
        total_counts = []
        for i in range(len(self.class_names)):
            class_mask = (y_true == i)
            class_errors = np.sum(error_mask & class_mask)
            class_total = np.sum(class_mask)
            error_counts.append(class_errors)
            total_counts.append(class_total)
        
        error_rates = [e/t*100 if t > 0 else 0 for e, t in zip(error_counts, total_counts)]
        
        bars = plt.bar(self.class_names, error_rates, color='red', alpha=0.7)
        plt.title('Tasa de Error por Clase (%)')
        plt.ylabel('Tasa de Error (%)')
        plt.xticks(rotation=45)
        
        # Agregar valores en las barras
        for bar, rate in zip(bars, error_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Análisis de errores guardado como: error_analysis.png")
    
    def _generate_detailed_report(self, test_loss, test_accuracy, precision, recall, f1, support,
                                confidence_analysis, overfitting_analysis, y_true, y_pred, y_pred_proba):
        """Genera un reporte detallado en texto"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
=== REPORTE DETALLADO DE ENTRENAMIENTO LSTM LSA ===
Fecha: {timestamp}

=== CONFIGURACIÓN DEL MODELO ===
- Arquitectura: LSTM con {len(self.class_names)} clases
- Secuencia máxima: {self.max_sequence_length} frames
- Keypoints por frame: {self.keypoints_length}
- Clases: {', '.join(self.class_names)}

=== MÉTRICAS GENERALES ===
- Pérdida de prueba: {test_loss:.4f}
- Precisión de prueba: {test_accuracy:.4f}
- Precisión macro: {np.mean(precision):.4f}
- Recall macro: {np.mean(recall):.4f}
- F1-Score macro: {np.mean(f1):.4f}

=== ANÁLISIS POR CLASE ===
"""
        for i, class_name in enumerate(self.class_names):
            report += f"- {class_name}:\n"
            report += f"  * Precisión: {precision[i]:.4f}\n"
            report += f"  * Recall: {recall[i]:.4f}\n"
            report += f"  * F1-Score: {f1[i]:.4f}\n"
            report += f"  * Muestras: {support[i]}\n"
        
        report += f"""
=== ANÁLISIS DE CONFIANZA ===
- Confianza promedio (correctas): {confidence_analysis['avg_correct_confidence']:.4f}
- Confianza promedio (incorrectas): {confidence_analysis['avg_incorrect_confidence']:.4f}
- Predicciones alta confianza (>0.9): {confidence_analysis['high_confidence_count']}/{len(y_true)}
- Predicciones baja confianza (<0.5): {confidence_analysis['low_confidence_count']}/{len(y_true)}

=== ANÁLISIS DE OVERFITTING ===
- Estado: {overfitting_analysis['status']}
- Gap de precisión: {overfitting_analysis['accuracy_gap']:.4f}
- Gap de pérdida: {overfitting_analysis['loss_gap']:.4f}
- Recomendación: {overfitting_analysis['recommendation']}

=== INTERPRETACIÓN DE RESULTADOS ===
"""
        
        # Interpretación automática
        if test_accuracy > 0.9:
            report += "✅ EXCELENTE: El modelo tiene muy buen rendimiento.\n"
        elif test_accuracy > 0.8:
            report += "✅ BUENO: El modelo tiene buen rendimiento.\n"
        elif test_accuracy > 0.7:
            report += "⚠️ REGULAR: El modelo necesita mejoras.\n"
        else:
            report += "❌ MALO: El modelo necesita reentrenamiento.\n"
        
        if overfitting_analysis['accuracy_gap'] > 0.1:
            report += "⚠️ OVERFITTING: El modelo memoriza en lugar de generalizar.\n"
            report += "   Soluciones: Más datos, más regularización, menos complejidad.\n"
        
        if confidence_analysis['low_confidence_count'] > len(y_true) * 0.2:
            report += "⚠️ BAJA CONFIANZA: Muchas predicciones inciertas.\n"
            report += "   Soluciones: Más datos de entrenamiento, mejor calidad de datos.\n"
        
        # Identificar clases problemáticas
        worst_class_idx = np.argmin(f1)
        worst_class = self.class_names[worst_class_idx]
        report += f"⚠️ CLASE PROBLEMÁTICA: '{worst_class}' (F1: {f1[worst_class_idx]:.4f})\n"
        report += f"   Necesita más datos de entrenamiento o mejor calidad.\n"
        
        # Guardar reporte
        with open('detailed_training_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Reporte detallado guardado como: detailed_training_report.txt")
        return report
    
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
        
        print(f"\n=== ARCHIVOS GENERADOS ===")
        print(f"📊 Análisis y Métricas:")
        print(f"- detailed_training_report.txt - Reporte completo con interpretación")
        print(f"- training_history_lstm.png - Historial de entrenamiento")
        print(f"- confusion_matrix_lstm.png - Matriz de confusión")
        print(f"- class_performance_analysis.png - Rendimiento por clase")
        print(f"- confidence_analysis.png - Distribución de confianza")
        print(f"- error_analysis.png - Análisis detallado de errores")
        print(f"")
        print(f"🤖 Modelos:")
        print(f"- models/lsa_lstm_model.h5 - Modelo final")
        print(f"- models/lsa_lstm_model_metadata.json - Metadatos")
        print(f"- models/best_lstm_model.h5 - Mejor modelo durante entrenamiento")
        
        print(f"\n=== CÓMO INTERPRETAR LOS RESULTADOS ===")
        print(f"1. 📋 Lee 'detailed_training_report.txt' para interpretación completa")
        print(f"2. 📈 Revisa 'training_history_lstm.png' para detectar overfitting")
        print(f"3. 🎯 Analiza 'confusion_matrix_lstm.png' para errores entre clases")
        print(f"4. 📊 Examina 'class_performance_analysis.png' para clases problemáticas")
        print(f"5. 🎲 Verifica 'confidence_analysis.png' para confianza del modelo")
        print(f"6. ❌ Estudia 'error_analysis.png' para patrones de errores")
        
        print(f"\n✅ ¡Entrenamiento completado exitosamente!")
        print(f"Siguiente paso: python app_holistic_simple.py")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
