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
    
    # Encontrar frases que ya existen (case-insensitive)
    already_exist = []
    truly_new = []
    
    # Crear un conjunto de frases existentes en minúsculas para comparación rápida
    existing_phrases_lower = {phrase.lower() for phrase in existing_phrases}
    
    for phrase in new_phrases:
        if phrase.lower() in existing_phrases_lower:
            # Encontrar la frase original (con la capitalización original)
            original_phrase = next((existing for existing in existing_phrases 
                                  if existing.lower() == phrase.lower()), phrase)
            already_exist.append(original_phrase)
        else:
            truly_new.append(phrase)
    
    # Combinar frases existentes y nuevas, eliminando duplicados case-insensitive
    all_phrases = []
    seen_lower = set()
    
    # Primero agregar las frases existentes, manteniendo solo una versión de cada una
    for phrase in existing_phrases:
        if phrase.lower() not in seen_lower:
            all_phrases.append(phrase)
            seen_lower.add(phrase.lower())
    
    # Luego agregar las frases nuevas que no existan
    for phrase in truly_new:
        if phrase.lower() not in seen_lower:
            all_phrases.append(phrase)
            seen_lower.add(phrase.lower())
    
    # Ordenar alfabéticamente (case-insensitive)
    all_phrases.sort(key=str.lower)
    
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
        input_count = len(read_new_phrases(input_file))
        
        # Leer el archivo final para obtener el conteo real
        final_phrases = read_existing_phrases("data/frases_comercio.csv")
        final_count = len(final_phrases)
        
        print("\n=== ESTADÍSTICAS ACTUALIZADAS ===")
        print(f"Frases en {input_file}: {input_count}")
        print(f"Frases únicas en data/frases_comercio.csv: {final_count}")
        
        print("\n=== RESULTADOS ===")
        print(f"Frases que ya existían: {len(existing)}")
        # if existing:
        #     print("Frases existentes:")
        #     for phrase in existing:
        #         print(f"  - {phrase}")
        
        print(f"\nFrases nuevas agregadas: {len(final_phrases)}")
        if final_phrases:
            print("Frases nuevas:")
            for phrase in final_phrases:
                print(f"  - {phrase}")
        
        if not existing and not final_phrases:
            print("No se encontraron frases para procesar.")
        
        print(f"\nEl archivo 'data/frases_comercio.csv' ha sido actualizado y ordenado alfabéticamente.")
        print(f"Total de frases únicas finales: {final_count}")
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")

if __name__ == "__main__":
    main() 