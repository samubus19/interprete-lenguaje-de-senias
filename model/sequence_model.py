import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LSASequenceModel:
    def __init__(self, max_sequence_length=30, feature_dim=378, num_classes=None):
        """
        Modelo de secuencias para reconocimiento de frases en LSA
        
        Args:
            max_sequence_length (int): Longitud máxima de secuencia
            feature_dim (int): Dimensión del vector de características
            num_classes (int): Número de clases (frases)
        """
        self.max_sequence_length = max_sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, model_type='lstm'):
        """
        Construye el modelo de secuencias
        
        Args:
            model_type (str): Tipo de modelo ('lstm', 'gru', 'transformer')
        """
        if self.num_classes is None:
            raise ValueError("num_classes debe ser especificado antes de construir el modelo")
            
        if model_type == 'lstm':
            self.model = self._build_lstm_model()
        elif model_type == 'gru':
            self.model = self._build_gru_model()
        elif model_type == 'transformer':
            self.model = self._build_transformer_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
        return self.model
    
    def _build_lstm_model(self):
        """Construye modelo basado en LSTM"""
        inputs = layers.Input(shape=(self.max_sequence_length, self.feature_dim))
        
        # Normalización de entrada
        x = layers.LayerNormalization()(inputs)
        
        # Capas LSTM bidireccionales
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(32, dropout=0.3))(x)
        
        # Capas densas
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de salida
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSA_LSTM_Model')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def _build_gru_model(self):
        """Construye modelo basado en GRU"""
        inputs = layers.Input(shape=(self.max_sequence_length, self.feature_dim))
        
        # Normalización de entrada
        x = layers.LayerNormalization()(inputs)
        
        # Capas GRU bidireccionales
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.3))(x)
        x = layers.Bidirectional(layers.GRU(32, dropout=0.3))(x)
        
        # Capas densas
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de salida
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSA_GRU_Model')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def _build_transformer_model(self):
        """Construye modelo basado en Transformer"""
        inputs = layers.Input(shape=(self.max_sequence_length, self.feature_dim))
        
        # Embedding posicional
        x = self._add_positional_encoding(inputs)
        
        # Capas de atención multi-cabeza
        for _ in range(3):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed Forward
            ff_output = layers.Dense(256, activation='relu')(x)
            ff_output = layers.Dropout(0.1)(ff_output)
            ff_output = layers.Dense(self.feature_dim)(ff_output)
            
            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Capas densas finales
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Capa de salida
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSA_Transformer_Model')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def _add_positional_encoding(self, inputs):
        """Añade encoding posicional para el transformer"""
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Crear encoding posicional
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                         -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        
        pos_encoding = tf.zeros((seq_len, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
            tf.sin(position * div_term)
        )
        
        if d_model % 2 == 0:
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
                tf.cos(position * div_term)
            )
        
        return inputs + pos_encoding[tf.newaxis, :, :]
    
    def prepare_data(self, dataset_path):
        """
        Prepara los datos para entrenamiento
        
        Args:
            dataset_path (str): Ruta al dataset
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val, class_names)
        """
        # Cargar dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            
        sequences = dataset['sequences']
        labels = dataset['labels']
        self.num_classes = len(dataset['metadata']['phrases'])
        self.phrase_to_id = dataset['phrase_to_id']
        self.id_to_phrase = dataset['id_to_phrase']
        
        # Padding de secuencias
        X = self._pad_sequences(sequences)
        
        # Convertir labels a one-hot
        y = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        
        # División train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Datos preparados:")
        print(f"- Entrenamiento: {X_train.shape[0]} secuencias")
        print(f"- Validación: {X_val.shape[0]} secuencias")
        print(f"- Forma de entrada: {X_train.shape[1:]}")
        print(f"- Número de clases: {self.num_classes}")
        
        return X_train, X_val, y_train, y_val, list(dataset['metadata']['phrases'])
    
    def _pad_sequences(self, sequences):
        """Aplica padding a las secuencias"""
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > self.max_sequence_length:
                # Truncar si es muy largo
                padded_seq = seq[:self.max_sequence_length]
            else:
                # Padding con ceros si es muy corto
                padding_length = self.max_sequence_length - len(seq)
                padding = np.zeros((padding_length, self.feature_dim))
                padded_seq = np.vstack([seq, padding])
                
            padded_sequences.append(padded_seq)
            
        return np.array(padded_sequences)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Entrena el modelo
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            
        Returns:
            History: Historial de entrenamiento
        """
        if self.model is None:
            raise ValueError("El modelo debe ser construido antes del entrenamiento")
            
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
                'models/best_sequence_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenamiento
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el modelo
        
        Args:
            X_test, y_test: Datos de prueba
            class_names (list): Nombres de las clases
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de la evaluación")
            
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Métricas
        test_loss, test_acc, test_top_k = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n=== EVALUACIÓN DEL MODELO ===")
        print(f"Pérdida de prueba: {test_loss:.4f}")
        print(f"Precisión de prueba: {test_acc:.4f}")
        print(f"Top-K Precisión: {test_top_k:.4f}")
        
        # Reporte de clasificación
        print(f"\n=== REPORTE DE CLASIFICACIÓN ===")
  
        unique_labels = sorted(list(set(y_true_classes) | set(y_pred_classes)))
        if len(class_names) >= len(unique_labels):
            target_names_filtered = [class_names[i] for i in unique_labels]
            print(classification_report(y_true_classes, y_pred_classes, 
                                        target_names=target_names_filtered, zero_division=0))
        else:
            print(classification_report(y_true_classes, y_pred_classes, zero_division=0))
        
        # Matriz de confusión
        self._plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_top_k_accuracy': test_top_k,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Grafica la matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Verificar que las dimensiones coincidan
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        if len(class_names) >= len(unique_labels):
            labels_filtered = [class_names[i] for i in unique_labels]
        else:
            labels_filtered = [f"Clase_{i}" for i in unique_labels]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_filtered, yticklabels=labels_filtered)
        plt.title('Matriz de Confusión - Modelo de Secuencias LSA')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_sequence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Matriz de confusión guardada como: confusion_matrix_sequence.png")
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento disponible")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precisión
        axes[0, 0].plot(self.history.history['accuracy'], label='Entrenamiento')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validación')
        axes[0, 0].set_title('Precisión del Modelo')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Precisión')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Pérdida
        axes[0, 1].plot(self.history.history['loss'], label='Entrenamiento')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validación')
        axes[0, 1].set_title('Pérdida del Modelo')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Pérdida')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-K Accuracy
        axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], label='Entrenamiento')
        axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validación')
        axes[1, 0].set_title('Top-K Precisión')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Top-K Precisión')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (si está disponible)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Tasa de Aprendizaje')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('training_history_sequence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Historial de entrenamiento guardado como: training_history_sequence.png")
    
    def predict_sequence(self, sequence, threshold=0.5):
        """
        Predice la clase de una secuencia
        
        Args:
            sequence (numpy.array): Secuencia de landmarks
            threshold (float): Umbral de confianza
            
        Returns:
            dict: Predicción con confianza
        """
        if self.model is None:
            raise ValueError("El modelo debe ser cargado antes de hacer predicciones")
            
        # Preparar secuencia
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        else:
            padding_length = self.max_sequence_length - len(sequence)
            padding = np.zeros((padding_length, self.feature_dim))
            sequence = np.vstack([sequence, padding])
            
        # Predicción
        sequence_input = np.expand_dims(sequence, axis=0)
        prediction = self.model.predict(sequence_input, verbose=0)[0]
        
        # Obtener clase con mayor probabilidad
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        if confidence >= threshold:
            phrase = self.id_to_phrase[predicted_class]
            return {
                'phrase': phrase,
                'confidence': float(confidence),
                'class_id': int(predicted_class),
                'all_probabilities': {self.id_to_phrase[i]: float(prob) 
                                    for i, prob in enumerate(prediction)}
            }
        else:
            return {
                'phrase': 'DESCONOCIDO',
                'confidence': float(confidence),
                'class_id': -1,
                'all_probabilities': {self.id_to_phrase[i]: float(prob) 
                                    for i, prob in enumerate(prediction)}
            }
    
    def save_model(self, model_path, metadata_path=None):
        """Guarda el modelo y metadatos"""
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
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'phrase_to_id': self.phrase_to_id,
            'id_to_phrase': self.id_to_phrase
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        print(f"Modelo guardado en: {model_path}")
        print(f"Metadatos guardados en: {metadata_path}")
    
    def load_model(self, model_path, metadata_path=None):
        """Carga el modelo y metadatos"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
            
        # Cargar modelo
        self.model = tf.keras.models.load_model(model_path)
        
        # Cargar metadatos
        if metadata_path is None:
            metadata_path = model_path.replace('.h5', '_metadata.json')
            
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            self.max_sequence_length = metadata['max_sequence_length']
            self.feature_dim = metadata['feature_dim']
            self.num_classes = metadata['num_classes']
            self.phrase_to_id = metadata['phrase_to_id']
            self.id_to_phrase = metadata['id_to_phrase']
            
            print(f"Modelo cargado desde: {model_path}")
            print(f"Metadatos cargados desde: {metadata_path}")
        else:
            print(f"Advertencia: No se encontraron metadatos en: {metadata_path}")
