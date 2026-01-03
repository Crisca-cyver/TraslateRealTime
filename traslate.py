import queue
import sys
import time
import random
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# --- Configuración ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # Tamaño del bloque de audio (en samples) a procesar por callback
THRESHOLD = 0.01   # Umbral de amplitud para detectar silencio (ajustar según micrófono)
SILENCE_DURATION = 1.0  # Segundos de silencio para considerar fin de frase
MODEL_SIZE = "tiny"     # "tiny", "base", "small", "medium", "large"
DEVICE = "cpu"          # "cuda" si tienes GPU Nvidia, sino "cpu"
COMPUTE_TYPE = "int8"   # "float16", "int8_float16", "int8"

# --- Inicialización ---
print("Cargando modelo Whisper...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Modelo cargado.")

translator = GoogleTranslator(source='en', target='es')

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Callback de audio que se ejecuta en un hilo separado."""
    if status:
        print(status, file=sys.stderr)
    # Copiar datos a la cola
    audio_queue.put(indata.flatten().copy())

def main():
    print(f"\nIniciando traducción en tiempo real (EN -> ES)...")
    print("Habla en Inglés. Presiona Ctrl+C para salir.\n")

    # Buffer acumulativo de audio
    audio_buffer = np.array([], dtype=np.float32)
    last_speak_time = time.time()
    is_speaking = False
    
    # Iniciar stream de audio
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            # Leer todo lo que haya en la queue
            try:
                while True:
                    chunk = audio_queue.get_nowait()
                    audio_buffer = np.concatenate((audio_buffer, chunk))
            except queue.Empty:
                pass

            # Si el buffer está vacío, esperar un poco
            if len(audio_buffer) == 0:
                sd.sleep(50)
                continue

            # Detectar volumen (RMS) del último bloque añadido (aprox)
            # Para simplificar, miramos el buffer entero reciente si es pequeño
            # O mejor, mirar los últimos N samples
            check_size = min(len(audio_buffer), BLOCK_SIZE * 4) 
            recent_audio = audio_buffer[-check_size:]
            rms = np.sqrt(np.mean(recent_audio**2))

            current_time = time.time()

            if rms > THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    print(f"Voz detectada (RMS: {rms:.4f})")
                last_speak_time = current_time
            else:
                 # Debug: Imprimir RMS ocasionalmente si no se habla
                 if random.random() < 0.05:
                     print(f"Silencio... (RMS: {rms:.4f})")
            
            # Si ha pasado X tiempo desde el último sonido fuerte, y tenemos algo de audio
            silence_time = current_time - last_speak_time
            
            # Procesar si hay silencio suficiente Y tenemos suficiente audio acumulado (> 0.5s)
            if silence_time > SILENCE_DURATION and len(audio_buffer) > SAMPLE_RATE * 0.5:
                # Procesar frase
                # print("Procesando frase...")
                
                # Transcribir
                segments, info = model.transcribe(audio_buffer, beam_size=5, language="en")
                
                full_text = ""
                for segment in segments:
                    full_text += segment.text + " "
                
                full_text = full_text.strip()
                
                if full_text:
                    try:
                        translated = translator.translate(full_text)
                        print(f"\nEN: {full_text}")
                        print(f"ES: {translated}\n")
                        print("-" * 30)

                        # Guardar en archivo
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open("translations.md", "a", encoding="utf-8") as f:
                            f.write(f"## {timestamp}\n")
                            f.write(f"**Original:** {full_text}\n\n")
                            f.write(f"**Traducción:** {translated}\n")
                            f.write("---\n\n")

                    except Exception as e:
                        print(f"Error traduciendo: {e}")
                
                # Resetear buffer y estado
                audio_buffer = np.array([], dtype=np.float32)
                is_speaking = False

            # Limpiar buffer si se hace demasiado grande sin silencios (e.g. ruido constante)
            # para evitar crash de memoria. Max 30 segundos.
            if len(audio_buffer) > SAMPLE_RATE * 30:
                print("Buffer lleno (ruido constante?), reiniciando...")
                audio_buffer = np.array([], dtype=np.float32)
                is_speaking = False

            sd.sleep(50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    except Exception as e:
        print(f"\nOcurrió un error: {e}")