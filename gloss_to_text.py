import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime

# Importaciones opcionales
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Advertencia: OpenAI no está disponible. Instala con: pip install openai")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Advertencia: Transformers no está disponible. Instala con: pip install transformers")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Advertencia: spaCy no está disponible. Instala con: pip install spacy")

class GlossToTextConverter:
    def __init__(self, method='openai', api_key=None):
        """
        Convertidor de glosas LSA a texto natural
        
        Args:
            method (str): Método a usar ('openai', 'transformers', 'rules')
            api_key (str): API key para OpenAI (si se usa ese método)
        """
        self.method = method
        self.api_key = api_key
        
        # Configurar según el método elegido
        if method == 'openai':
            if not OPENAI_AVAILABLE:
                print("OpenAI no disponible, cambiando a método 'rules'")
                self.method = 'rules'
                self._setup_rules()
            else:
                self._setup_openai()
        elif method == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers no disponible, cambiando a método 'rules'")
                self.method = 'rules'
                self._setup_rules()
            else:
                self._setup_transformers()
        elif method == 'rules':
            self._setup_rules()
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def _setup_openai(self):
        """Configurar OpenAI API"""
        if not self.api_key:
            # Intentar obtener de variable de entorno
            self.api_key = os.getenv('OPENAI_API_KEY')
            
        if not self.api_key:
            print("Advertencia: No se encontró API key de OpenAI.")
            print("Configura la variable de entorno OPENAI_API_KEY o pásala como parámetro.")
            return
            
        openai.api_key = self.api_key
        
        # Prompt base para LSA
        self.lsa_prompt = """
Eres un experto en Lengua de Señas Argentina (LSA) y procesamiento de lenguaje natural.

Tu tarea es convertir secuencias de glosas (palabras individuales en LSA) a texto natural en español argentino.

Las glosas en LSA siguen estas características:
- Orden de palabras diferente al español hablado
- Sin artículos, preposiciones o conectores explícitos
- Estructura más directa y visual
- Tiempo verbal implícito en el contexto

Ejemplos:
- Glosas: "YO COMPRAR CARNE" → Texto: "Yo voy a comprar carne"
- Glosas: "MAMÁ COCINAR COMIDA RICA" → Texto: "Mamá cocina comida rica"
- Glosas: "MAÑANA TRABAJO IR" → Texto: "Mañana voy a trabajar"
- Glosas: "HERMANO PELOTA JUGAR" → Texto: "Mi hermano juega a la pelota"

Convierte las siguientes glosas a texto natural:
Glosas: "{gloss_sequence}"
Texto:"""
    
    def _setup_transformers(self):
        """Configurar modelo de Transformers local"""
        try:
            # Usar un modelo multilingüe para español
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.translator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
            print("Modelo de Transformers cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar modelo de Transformers: {e}")
            print("Instalando dependencias necesarias...")
            os.system("pip install transformers torch")
    
    def _setup_rules(self):
        """Configurar sistema basado en reglas"""
        if SPACY_AVAILABLE:
            try:
                # Cargar spaCy para español
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                print("Modelo de spaCy no encontrado. Usando reglas básicas sin spaCy.")
                self.nlp = None
        else:
            print("spaCy no disponible. Usando reglas básicas.")
            self.nlp = None
        
        # Diccionario de transformaciones comunes en LSA
        self.lsa_transformations = {
            # Pronombres
            'YO': 'yo',
            'TU': 'tú',
            'EL': 'él',
            'ELLA': 'ella',
            'NOSOTROS': 'nosotros',
            'USTEDES': 'ustedes',
            'ELLOS': 'ellos',
            
            # Verbos comunes con conjugación
            'COMPRAR': ['compro', 'compras', 'compra', 'compramos', 'compran'],
            'COMER': ['como', 'comes', 'come', 'comemos', 'comen'],
            'TRABAJAR': ['trabajo', 'trabajas', 'trabaja', 'trabajamos', 'trabajan'],
            'IR': ['voy', 'vas', 'va', 'vamos', 'van'],
            'VENIR': ['vengo', 'vienes', 'viene', 'venimos', 'vienen'],
            'HACER': ['hago', 'haces', 'hace', 'hacemos', 'hacen'],
            'TENER': ['tengo', 'tienes', 'tiene', 'tenemos', 'tienen'],
            'SER': ['soy', 'eres', 'es', 'somos', 'son'],
            'ESTAR': ['estoy', 'estás', 'está', 'estamos', 'están'],
            
            # Sustantivos con artículos
            'CASA': 'la casa',
            'TRABAJO': 'el trabajo',
            'COMIDA': 'la comida',
            'AGUA': 'el agua',
            'CARNE': 'la carne',
            'PELOTA': 'la pelota',
            'LIBRO': 'el libro',
            'ESCUELA': 'la escuela',
            
            # Adjetivos
            'BUENO': 'bueno',
            'MALO': 'malo',
            'GRANDE': 'grande',
            'PEQUEÑO': 'pequeño',
            'RICO': 'rico',
            'FEO': 'feo',
            'BONITO': 'bonito',
            
            # Tiempo
            'HOY': 'hoy',
            'MAÑANA': 'mañana',
            'AYER': 'ayer',
            'AHORA': 'ahora',
            'DESPUÉS': 'después',
            'ANTES': 'antes',
            
            # Familia
            'MAMÁ': 'mamá',
            'PAPÁ': 'papá',
            'HERMANO': 'mi hermano',
            'HERMANA': 'mi hermana',
            'HIJO': 'mi hijo',
            'HIJA': 'mi hija',
        }
        
        # Patrones de estructura LSA
        self.lsa_patterns = [
            # SUJETO + VERBO + OBJETO
            (r'(\w+)\s+(\w+)\s+(\w+)$', self._pattern_svo),
            # TIEMPO + SUJETO + VERBO
            (r'(\w+)\s+(\w+)\s+(\w+)$', self._pattern_tsv),
            # SUJETO + OBJETO + VERBO
            (r'(\w+)\s+(\w+)\s+(\w+)$', self._pattern_sov),
        ]
    
    def convert_gloss_to_text(self, gloss_sequence: str, context: Optional[str] = None) -> Dict:
        """
        Convierte una secuencia de glosas a texto natural
        
        Args:
            gloss_sequence (str): Secuencia de glosas separadas por espacios
            context (str): Contexto adicional (opcional)
            
        Returns:
            dict: Resultado con texto convertido y metadatos
        """
        if not gloss_sequence or not gloss_sequence.strip():
            return {
                'original_gloss': gloss_sequence,
                'converted_text': '',
                'confidence': 0.0,
                'method': self.method,
                'error': 'Secuencia de glosas vacía'
            }
        
        try:
            if self.method == 'openai':
                result = self._convert_with_openai(gloss_sequence, context)
            elif self.method == 'transformers':
                result = self._convert_with_transformers(gloss_sequence, context)
            elif self.method == 'rules':
                result = self._convert_with_rules(gloss_sequence, context)
            else:
                raise ValueError(f"Método no implementado: {self.method}")
                
            return result
            
        except Exception as e:
            return {
                'original_gloss': gloss_sequence,
                'converted_text': gloss_sequence.lower(),  # Fallback básico
                'confidence': 0.1,
                'method': self.method,
                'error': str(e)
            }
    
    def _convert_with_openai(self, gloss_sequence: str, context: Optional[str] = None) -> Dict:
        """Conversión usando OpenAI API"""
        if not self.api_key:
            raise ValueError("API key de OpenAI no configurada")
        
        prompt = self.lsa_prompt.format(gloss_sequence=gloss_sequence)
        
        if context:
            prompt += f"\nContexto adicional: {context}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en LSA y procesamiento de lenguaje natural."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            converted_text = response.choices[0].message.content.strip()
            
            return {
                'original_gloss': gloss_sequence,
                'converted_text': converted_text,
                'confidence': 0.9,
                'method': 'openai',
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            raise Exception(f"Error en OpenAI API: {e}")
    
    def _convert_with_transformers(self, gloss_sequence: str, context: Optional[str] = None) -> Dict:
        """Conversión usando modelo de Transformers local"""
        # Preparar input para el modelo
        input_text = f"Convertir glosas LSA a español: {gloss_sequence}"
        
        if context:
            input_text += f" Contexto: {context}"
        
        try:
            # Generar texto
            result = self.translator(input_text)
            converted_text = result[0]['generated_text']
            
            # Limpiar el resultado
            converted_text = self._clean_generated_text(converted_text, gloss_sequence)
            
            return {
                'original_gloss': gloss_sequence,
                'converted_text': converted_text,
                'confidence': 0.7,
                'method': 'transformers'
            }
            
        except Exception as e:
            raise Exception(f"Error en modelo Transformers: {e}")
    
    def _convert_with_rules(self, gloss_sequence: str, context: Optional[str] = None) -> Dict:
        """Conversión usando sistema basado en reglas"""
        words = gloss_sequence.upper().split()
        
        # Aplicar transformaciones básicas
        converted_words = []
        for word in words:
            if word in self.lsa_transformations:
                transformation = self.lsa_transformations[word]
                if isinstance(transformation, list):
                    # Para verbos, usar primera persona por defecto
                    converted_words.append(transformation[0])
                else:
                    converted_words.append(transformation)
            else:
                converted_words.append(word.lower())
        
        # Aplicar patrones estructurales
        converted_text = ' '.join(converted_words)
        
        # Mejorar con reglas gramaticales
        converted_text = self._apply_grammar_rules(converted_text, words)
        
        return {
            'original_gloss': gloss_sequence,
            'converted_text': converted_text,
            'confidence': 0.6,
            'method': 'rules'
        }
    
    def _apply_grammar_rules(self, text: str, original_words: List[str]) -> str:
        """Aplica reglas gramaticales básicas"""
        # Capitalizar primera letra
        text = text.capitalize()
        
        # Detectar patrones comunes
        if len(original_words) >= 3:
            # Patrón: PRONOMBRE + VERBO + OBJETO
            if original_words[0] in ['YO', 'TU', 'EL', 'ELLA']:
                # Agregar conectores si es necesario
                if 'COMPRAR' in original_words or 'IR' in original_words:
                    text = text.replace(' compro ', ' voy a comprar ')
                    text = text.replace(' ir ', ' voy a ir ')
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _clean_generated_text(self, generated_text: str, original_gloss: str) -> str:
        """Limpia el texto generado por modelos"""
        # Remover prefijos comunes de los modelos
        prefixes_to_remove = [
            "Convertir glosas LSA a español:",
            "Texto:",
            "Resultado:",
            original_gloss
        ]
        
        for prefix in prefixes_to_remove:
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):].strip()
        
        # Limpiar y capitalizar
        generated_text = generated_text.strip()
        if generated_text:
            generated_text = generated_text[0].upper() + generated_text[1:]
        
        return generated_text
    
    def convert_multiple_glosses(self, gloss_list: List[str], context: Optional[str] = None) -> List[Dict]:
        """
        Convierte múltiples secuencias de glosas
        
        Args:
            gloss_list (list): Lista de secuencias de glosas
            context (str): Contexto adicional
            
        Returns:
            list: Lista de resultados de conversión
        """
        results = []
        
        for i, gloss_sequence in enumerate(gloss_list):
            print(f"Procesando glosa {i+1}/{len(gloss_list)}: {gloss_sequence}")
            result = self.convert_gloss_to_text(gloss_sequence, context)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Guarda los resultados en un archivo JSON"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'method': self.method,
            'total_conversions': len(results),
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"Resultados guardados en: {output_file}")


def main():
    """Función principal para probar el convertidor"""
    print("=== CONVERTIDOR DE GLOSAS LSA A TEXTO NATURAL ===")
    
    # Ejemplos de glosas para probar
    test_glosses = [
        "YO COMPRAR CARNE",
        "MAMÁ COCINAR COMIDA RICA",
        "MAÑANA TRABAJO IR",
        "HERMANO PELOTA JUGAR",
        "HOY ESCUELA VENIR",
        "PAPÁ CASA LLEGAR"
    ]
    
    print("\nMétodos disponibles:")
    print("1. OpenAI API (requiere API key)")
    print("2. Transformers (modelo local)")
    print("3. Reglas gramaticales (básico)")
    
    method_choice = input("\nSelecciona método (1-3): ").strip()
    
    if method_choice == '1':
        api_key = input("Ingresa tu API key de OpenAI (o presiona Enter para usar variable de entorno): ").strip()
        converter = GlossToTextConverter(method='openai', api_key=api_key if api_key else None)
    elif method_choice == '2':
        converter = GlossToTextConverter(method='transformers')
    else:
        converter = GlossToTextConverter(method='rules')
    
    print(f"\n=== PROBANDO CONVERSIONES ===")
    results = converter.convert_multiple_glosses(test_glosses)
    
    print(f"\n=== RESULTADOS ===")
    for result in results:
        print(f"Glosas: {result['original_gloss']}")
        print(f"Texto: {result['converted_text']}")
        print(f"Confianza: {result['confidence']:.2f}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        print("-" * 50)
    
    # Guardar resultados
    converter.save_results(results, 'gloss_conversion_results.json')
    
    print("\n¡Conversión completada!")


if __name__ == "__main__":
    main()
