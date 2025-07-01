import csv
import os
from typing import List, Tuple

def read_existing_phrases(csv_file: str) -> List[str]:
    """
    Lee las frases existentes del archivo CSV.
    
    Args:
        csv_file: Ruta al archivo CSV existente
        
    Returns:
        Lista de frases existentes
    """
    existing_phrases = []
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_phrases.append(row['frase'].strip())
    
    return existing_phrases

def read_new_phrases(input_csv: str) -> List[str]:
    """
    Lee las nuevas frases del archivo CSV de entrada.
    
    Args:
        input_csv: Ruta al archivo CSV con nuevas frases
        
    Returns:
        Lista de nuevas frases
    """
    new_phrases = []
    
    if not os.path.exists(input_csv):
        print(f"Error: El archivo {input_csv} no existe.")
        return new_phrases
    
    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Verificar que la fila no esté vacía
                # Procesar todas las columnas de la fila
                for cell in row:
                    phrase = cell.strip()
                    if phrase and phrase not in new_phrases:
                        new_phrases.append(phrase)
    
    return new_phrases

def process_phrases(input_csv: str, output_csv: str = "data/frases_comercio.csv") -> Tuple[List[str], List[str]]:
    """
    Procesa las frases del archivo de entrada y las compara con el archivo existente.
    
    Args:
        input_csv: Ruta al archivo CSV con nuevas frases
        output_csv: Ruta al archivo CSV de salida (por defecto frases_comercio.csv)
        
    Returns:
        Tupla con (frases_existentes, frases_nuevas)
    """
    # Leer frases existentes
    existing_phrases = read_existing_phrases(output_csv)
    print("\n=== ESTADÍSTICAS DE ARCHIVOS ===")
    print(f"Frases existentes encontradas: {len(existing_phrases)}")
    
    # Leer nuevas frases
    new_phrases = read_new_phrases(input_csv)
    print(f"Nuevas frases encontradas: {len(new_phrases)}")
    
    # Encontrar frases que ya existen
    already_exist = []
    truly_new = []
    
    for phrase in new_phrases:
        if phrase in existing_phrases:
            already_exist.append(phrase)
        else:
            truly_new.append(phrase)
    
    # Combinar todas las frases y ordenar alfabéticamente
    all_phrases = existing_phrases + truly_new
    all_phrases.sort(key=str.lower)  # Ordenamiento case-insensitive
    
    # Escribir el archivo actualizado
    write_phrases_to_csv(all_phrases, output_csv)
    
    return already_exist, truly_new

def write_phrases_to_csv(phrases: List[str], csv_file: str):
    """
    Escribe las frases al archivo CSV con la estructura requerida.
    
    Args:
        phrases: Lista de frases a escribir
        csv_file: Ruta al archivo CSV de salida
    """
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Escribir encabezado
        writer.writerow(['frase', 'glosa'])
        
        # Escribir frases
        for phrase in phrases:
            writer.writerow([phrase, ''])  # Columna glosa vacía
    
    print(f"Archivo actualizado: {csv_file}")

def main():
    """
    Función principal que ejecuta el procesamiento de frases.
    """
    print("=== Procesador de Frases para Lenguaje de Señas ===")
    
    # Archivo de entrada por defecto
    input_file = "data/frases_entrada.csv"
    
    print(f"Procesando archivo: {input_file}")
    
    # Procesar las frases
    try:
        existing, new = process_phrases(input_file)
        
        # Contar frases en cada archivo
        existing_count = len(read_existing_phrases("data/frases_comercio.csv"))
        input_count = len(read_new_phrases(input_file))
        
        print("\n=== ESTADÍSTICAS ACTUALIZADAS ===")
        print(f"Frases en {input_file}: {input_count}")
        print(f"Frases en data/frases_comercio.csv: {existing_count}")
        
        print("\n=== RESULTADOS ===")
        print(f"Frases que ya existían: {len(existing)}")
        # if existing:
        #     print("Frases existentes:")
        #     for phrase in existing:
        #         print(f"  - {phrase}")
        
        print(f"\nFrases nuevas agregadas: {len(new)}")
        if new:
            print("Frases nuevas:")
            for phrase in new:
                print(f"  - {phrase}")
        
        if not existing and not new:
            print("No se encontraron frases para procesar.")
        
        print(f"\nEl archivo 'data/frases_comercio.csv' ha sido actualizado y ordenado alfabéticamente.")
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")

if __name__ == "__main__":
    main() 