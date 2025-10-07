import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import random

class LSADataAugmentation:
    def __init__(self):
        """Aumenta artificialmente el dataset de LSA"""
        pass
    
    def add_noise(self, sequence, noise_factor=0.01):
        """Agrega ruido gaussiano a la secuencia"""
        noise = np.random.normal(0, noise_factor, sequence.shape)
        return sequence + noise
    
    def time_shift(self, sequence, shift_range=3):
        """Desplaza temporalmente la secuencia"""
        shift = random.randint(-shift_range, shift_range)
        if shift > 0:
            # Pad al inicio, truncar al final
            padded = np.zeros((shift, sequence.shape[1]))
            shifted = np.vstack([padded, sequence[:-shift]])
        elif shift < 0:
            # Truncar al inicio, pad al final
            padded = np.zeros((-shift, sequence.shape[1]))
            shifted = np.vstack([sequence[-shift:], padded])
        else:
            shifted = sequence
        return shifted
    
    def speed_variation(self, sequence, speed_factor_range=(0.8, 1.2)):
        """Varía la velocidad de la secuencia"""
        speed_factor = random.uniform(*speed_factor_range)
        
        # Cambiar la longitud temporal
        original_length = len(sequence)
        new_length = int(original_length * speed_factor)
        
        # Interpolar para mantener 30 frames
        indices = np.linspace(0, original_length - 1, new_length)
        
        # Interpolar cada keypoint
        interpolated = []
        for i in range(sequence.shape[1]):
            interp_values = np.interp(
                np.linspace(0, new_length - 1, 30),
                np.arange(new_length),
                np.interp(indices, np.arange(original_length), sequence[:, i])
            )
            interpolated.append(interp_values)
        
        return np.array(interpolated).T
    
    def spatial_transform(self, sequence, scale_range=(0.9, 1.1), 
                         rotation_range=(-5, 5), translation_range=(-0.05, 0.05)):
        """Aplica transformaciones espaciales"""
        # Solo transformar coordenadas x, y (no z ni visibility)
        transformed = sequence.copy()
        
        # Parámetros de transformación
        scale = random.uniform(*scale_range)
        rotation = np.radians(random.uniform(*rotation_range))
        tx = random.uniform(*translation_range)
        ty = random.uniform(*translation_range)
        
        # Matriz de rotación
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        
        # Transformar pose landmarks (x, y cada 4 valores)
        for i in range(0, 132, 4):  # 33 puntos * 4 valores
            x, y = transformed[:, i], transformed[:, i + 1]
            
            # Aplicar transformaciones
            x_new = scale * (cos_r * x - sin_r * y) + tx
            y_new = scale * (sin_r * x + cos_r * y) + ty
            
            transformed[:, i] = x_new
            transformed[:, i + 1] = y_new
        
        # Transformar face landmarks (x, y, z cada 3 valores)
        for i in range(132, 132 + 1404, 3):  # 468 puntos * 3 valores
            x, y = transformed[:, i], transformed[:, i + 1]
            
            x_new = scale * (cos_r * x - sin_r * y) + tx
            y_new = scale * (sin_r * x + cos_r * y) + ty
            
            transformed[:, i] = x_new
            transformed[:, i + 1] = y_new
        
        # Transformar hand landmarks
        for i in range(132 + 1404, 132 + 1404 + 126, 3):  # 42 puntos * 3 valores
            x, y = transformed[:, i], transformed[:, i + 1]
            
            x_new = scale * (cos_r * x - sin_r * y) + tx
            y_new = scale * (sin_r * x + cos_r * y) + ty
            
            transformed[:, i] = x_new
            transformed[:, i + 1] = y_new
        
        return transformed
    
    def augment_sequence(self, sequence, num_augmentations=5):
        """Genera múltiples versiones aumentadas de una secuencia"""
        augmented = [sequence]  # Original
        
        for _ in range(num_augmentations):
            aug_seq = sequence.copy()
            
            # Aplicar transformaciones aleatoriamente
            if random.random() > 0.5:
                aug_seq = self.add_noise(aug_seq)
            
            if random.random() > 0.5:
                aug_seq = self.time_shift(aug_seq)
            
            if random.random() > 0.5:
                aug_seq = self.speed_variation(aug_seq)
            
            if random.random() > 0.5:
                aug_seq = self.spatial_transform(aug_seq)
            
            augmented.append(aug_seq)
        
        return augmented
    
    def augment_dataset(self, dataset_path, output_path, augmentation_factor=10):
        """
        Aumenta todo el dataset
        
        Args:
            dataset_path: Ruta al dataset original
            output_path: Ruta para guardar dataset aumentado
            augmentation_factor: Factor de aumento (10x = 10 veces más datos)
        """
        print(f"=== AUMENTANDO DATASET ===")
        print(f"Factor de aumento: {augmentation_factor}x")
        
        # Cargar dataset original
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        X_original = dataset['X']
        y_original = dataset['y']
        
        print(f"Dataset original: {X_original.shape}")
        
        # Listas para datos aumentados
        X_augmented = []
        y_augmented = []
        
        # Procesar cada secuencia
        for i, (sequence, label) in enumerate(zip(X_original, y_original)):
            # Agregar original
            X_augmented.append(sequence)
            y_augmented.append(label)
            
            # Generar versiones aumentadas
            augmented_sequences = self.augment_sequence(
                sequence, num_augmentations=augmentation_factor-1
            )
            
            # Agregar aumentadas (excluyendo la original que ya agregamos)
            for aug_seq in augmented_sequences[1:]:
                X_augmented.append(aug_seq)
                y_augmented.append(label)
            
            print(f"  Procesada secuencia {i+1}/{len(X_original)} - Clase: {dataset['class_names'][label]}")
        
        # Convertir a numpy arrays
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        print(f"Dataset aumentado: {X_augmented.shape}")
        print(f"Aumento: {len(X_augmented) / len(X_original):.1f}x")
        
        # Crear nuevo dataset
        augmented_dataset = {
            'X': X_augmented,
            'y': y_augmented,
            'class_names': dataset['class_names'],
            'max_sequence_length': dataset['max_sequence_length'],
            'keypoints_length': dataset['keypoints_length'],
            'num_classes': dataset['num_classes'],
            'augmented': True,
            'augmentation_factor': augmentation_factor
        }
        
        # Guardar dataset aumentado
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(augmented_dataset, f)
        
        print(f"Dataset aumentado guardado en: {output_path}")
        
        # Mostrar distribución por clase
        unique, counts = np.unique(y_augmented, return_counts=True)
        print(f"\nDistribución por clase:")
        for class_id, count in zip(unique, counts):
            class_name = dataset['class_names'][class_id]
            print(f"  {class_name}: {count} secuencias")
        
        return augmented_dataset

def main():
    """Función principal para aumentar datos"""
    augmenter = LSADataAugmentation()
    
    # Verificar que existe el dataset original
    original_dataset = "models/sequence_dataset.pkl"
    if not os.path.exists(original_dataset):
        print(f"❌ No se encontró el dataset original: {original_dataset}")
        print("Ejecuta primero: python sequence_data_processor.py")
        return
    
    # Aumentar dataset
    augmented_dataset = augmenter.augment_dataset(
        dataset_path=original_dataset,
        output_path="models/sequence_dataset_augmented.pkl",
        augmentation_factor=20  # 20x más datos
    )
    
    print(f"\n✅ Dataset aumentado exitosamente!")
    print(f"Datos originales: 25 secuencias")
    print(f"Datos aumentados: {len(augmented_dataset['X'])} secuencias")
    print(f"Siguiente paso: python train_lstm_model.py")

if __name__ == "__main__":
    main()
