import os
import sys
import argparse
from model.sequence_model import LSASequenceModel

def main():
    """Función principal para entrenar el modelo de secuencias LSA"""
    parser = argparse.ArgumentParser(description='Entrenar modelo de secuencias para LSA')
    parser.add_argument('--dataset', type=str, default='dataset/lsa_sequence_dataset.pkl',
                       help='Ruta al dataset de secuencias')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'transformer'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch')
    parser.add_argument('--max_seq_length', type=int, default=30,
                       help='Longitud máxima de secuencia')
    
    args = parser.parse_args()
    
    print("=== ENTRENAMIENTO DE MODELO DE SECUENCIAS LSA ===")
    print(f"Dataset: {args.dataset}")
    print(f"Tipo de modelo: {args.model_type}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Longitud máxima de secuencia: {args.max_seq_length}")
    
    # Verificar que existe el dataset
    if not os.path.exists(args.dataset):
        print(f"\nError: No se encontró el dataset en {args.dataset}")
        print("Ejecuta los siguientes scripts en orden:")
        print("1. python frame_extractor.py")
        print("2. python landmark_extractor.py")
        return
    
    # Crear modelo
    model = LSASequenceModel(max_sequence_length=args.max_seq_length)
    
    # Preparar datos
    print("\n=== PREPARANDO DATOS ===")
    X_train, X_val, y_train, y_val, class_names = model.prepare_data(args.dataset)
    
    # Construir modelo
    print(f"\n=== CONSTRUYENDO MODELO {args.model_type.upper()} ===")
    model.build_model(model_type=args.model_type)
    
    print("\nArquitectura del modelo:")
    model.model.summary()
    
    # Entrenar modelo
    print(f"\n=== ENTRENANDO MODELO ===")
    history = model.train(X_train, y_train, X_val, y_val, 
                         epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluar modelo
    print(f"\n=== EVALUANDO MODELO ===")
    results = model.evaluate(X_val, y_val, class_names)
    
    # Graficar resultados
    print(f"\n=== GENERANDO GRÁFICOS ===")
    model.plot_training_history()
    
    # Guardar modelo
    print(f"\n=== GUARDANDO MODELO ===")
    os.makedirs('models', exist_ok=True)
    model_path = f'models/lsa_sequence_model_{args.model_type}.h5'
    model.save_model(model_path)
    
    # Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Modelo entrenado: {args.model_type}")
    print(f"Precisión final: {results['test_accuracy']:.4f}")
    print(f"Top-K Precisión: {results['test_top_k_accuracy']:.4f}")
    print(f"Número de clases: {len(class_names)}")
    print(f"Clases: {', '.join(class_names)}")
    print(f"\nArchivos generados:")
    print(f"- Modelo: {model_path}")
    print(f"- Metadatos: {model_path.replace('.h5', '_metadata.json')}")
    print(f"- Historial: training_history_sequence.png")
    print(f"- Matriz de confusión: confusion_matrix_sequence.png")
    
    print(f"\n¡Entrenamiento completado exitosamente!")
    print(f"Siguiente paso: Implementar el módulo de conversión glosas → texto natural")

if __name__ == "__main__":
    main()