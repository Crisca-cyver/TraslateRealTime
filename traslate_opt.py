import queue
import sys
import time
import random
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import argparse
import configparser
from collections import deque
import os
import logging

# --- Configuración por defecto ---
DEFAULTS = {
    'SAMPLE_RATE': 16000,
    'BLOCK_SIZE': 4000,
    'THRESHOLD': 0.01,
    'SILENCE_DURATION': 1.0,
    'MODEL_SIZE': 'tiny',
    'DEVICE': 'cpu',
    'COMPUTE_TYPE': 'int8',
    'SOURCE_LANG': 'en',
    'TARGET_LANG': 'es',
    'BUFFER_SECONDS': 30
}

# --- Cargar configuración desde archivo opcional ---
def load_config_file(config_path="config.ini"):
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
        params = {}
        for key in DEFAULTS.keys():
            try:
                params[key] = type(DEFAULTS[key])(config['DEFAULT'].get(key, DEFAULTS[key]))
            except Exception:
                params[key] = DEFAULTS[key]
        return params
    else:
        return DEFAULTS.copy()

# --- Argumentos de línea de comando ---
def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Speech Translator (Whisper + GoogleTranslator)")
    parser.add_argument('--model_size', default=None, help="Modelo Whisper (tiny, base, small, medium, large)")
    parser.add_argument('--device', default=None, help="Dispositivo (cpu o cuda)")
    parser.add_argument('--compute_type', default=None, help="Tipo de cómputo para Whisper (float16, int8_float16, int8)")
    parser.add_argument('--source_lang', default=None, help="Idioma original (ej: en)")
    parser.add_argument('--target_lang', default=None, help="Idioma de destino (ej: es)")
    parser.add_argument('--threshold', type=float, default=None, help="Umbral de detección de silencio [0.0-1.0]")
    parser.add_argument('--sample_rate', type=int, default=None, help="Sample rate de entrada de audio")
    parser.add_argument('--block_size', type=int, default=None, help="Bloque de samples por callback")
    parser.add_argument('--silence_duration', type=float, default=None, help="Segundos de silencio para corte de frase")
    parser.add_argument('--config', default='config.ini', help='Archivo de configuración opcional')
    return parser.parse_args()

# --- Logger Avanzado ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                    handlers=[logging.StreamHandler(), logging.FileHandler("traslate.log", encoding='utf-8')])
logger = logging.getLogger('Traslate')

# --- Volumen en consola ---
def print_vol_bar(rms, threshold):
    bar_length = 40
    filled = int(min(rms/threshold, 1.0) * bar_length)
    bar = ('#' * filled).ljust(bar_length)
    sys.stdout.write(f"\rVolumen: [{bar}] {rms:.4f}   ")
    sys.stdout.flush()

# --- Selección de dispositivo ---
def choose_input_device():
    print("\n== Dispositivos de Entrada de Audio ==")
    devices = sd.query_devices()
    input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
    for idx in input_devices:
        info = devices[idx]
        print(f"{idx}: {info['name']}")
    try:
        selected = int(input("Selecciona número de dispositivo (ENTER para predeterminado): ") or -1)
        if selected in input_devices:
            return selected
    except Exception:
        pass
    return None  # usa por defecto

# --- Main ---
def main():
    # --- Cargar configuración ---
    config_file = parse_args().config
    params = load_config_file(config_file)
    args = parse_args()
    # Sobre-escribir con argumentos si se proveen
    for k in DEFAULTS:
        user_arg = getattr(args, k.lower(), None)
        if user_arg is not None:
            params[k] = user_arg

    # Mostrar parámetros en uso
    logger.info(f"== Parámetros de traducción ==\n{params}")
    print(f"Guardando logs en traslate.log. Traducciones en translations.md")

    logger.info("Cargando modelo Whisper...")
    model = WhisperModel(params['MODEL_SIZE'], device=params['DEVICE'], compute_type=params['COMPUTE_TYPE'])
    logger.info("Modelo cargado.")

    translator = GoogleTranslator(source=params['SOURCE_LANG'], target=params['TARGET_LANG'])
    audio_queue = queue.Queue()
    buffer_samples = int(params['SAMPLE_RATE'] * params.get('BUFFER_SECONDS', 30))
    # deque para buffer eficiente, maxlen=buffer_samples
    audio_buffer = deque(maxlen=buffer_samples)
    last_speak_time = time.time()
    is_speaking = False

    # Selección de input device
    device = choose_input_device()

    def callback(indata, frames, _time, status):
        if status:
            logger.warning(f"Estado en callback: {status}")
        audio_queue.put(indata.flatten().copy())

    print(f"\nIniciando traducción en tiempo real ({params['SOURCE_LANG'].upper()} -> {params['TARGET_LANG'].upper()})...")
    print("Habla. Presiona Ctrl+C para salir.\n")

    # --- Audio Stream ---
    with sd.InputStream(callback=callback, channels=1, samplerate=params['SAMPLE_RATE'], blocksize=params['BLOCK_SIZE'], device=device):
        try:
            while True:
                # Vaciar toda la queue
                try:
                    while True:
                        chunk = audio_queue.get_nowait()
                        audio_buffer.extend(chunk)
                except queue.Empty:
                    pass

                # Nada de audio, esperar
                if len(audio_buffer) == 0:
                    sd.sleep(50)
                    continue

                # Volumen reciente
                check_size = min(len(audio_buffer), params['BLOCK_SIZE'] * 4)
                recent_audio = np.array([audio_buffer[-i-1] for i in range(check_size)][::-1], dtype=np.float32)
                if len(recent_audio) == 0:
                    continue

                rms = np.sqrt(np.mean(recent_audio ** 2))
                print_vol_bar(rms, params['THRESHOLD'])

                current_time = time.time()
                if rms > params['THRESHOLD']:
                    if not is_speaking:
                        is_speaking = True
                        logger.info(f"Voz detectada (RMS: {rms:.4f})")
                    last_speak_time = current_time
                else:
                    if random.random() < 0.05:
                        logger.debug(f"Silencio... (RMS: {rms:.4f})")

                # Silencio suficiente + audio acumulado
                silence_time = current_time - last_speak_time
                min_len = int(params['SAMPLE_RATE'] * 0.5)
                if silence_time > params['SILENCE_DURATION'] and len(audio_buffer) > min_len:
                    # Procesar frase completa
                    phrase = np.array(audio_buffer, dtype=np.float32)
                    try:
                        segments, info = model.transcribe(phrase, beam_size=5, language=params['SOURCE_LANG'])
                        full_text = " ".join([s.text for s in segments]).strip()
                        if full_text:
                            try:
                                translated = translator.translate(full_text)
                                print(f"\nEN: {full_text}\n{params['TARGET_LANG'].upper()}: {translated}\n{'-'*30}")
                                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                                with open("translations.md", "a", encoding="utf-8") as f:
                                    f.write(f"## {timestamp}\n")
                                    f.write(f"**Original ({params['SOURCE_LANG']}):** {full_text}\n\n")
                                    f.write(f"**Traducción ({params['TARGET_LANG']}):** {translated}\n")
                                    f.write("---\n\n")
                                logger.info(f"Texto traducido y guardado [{timestamp}]")
                            except Exception as e:
                                logger.error(f"Error en traducción: {e}")
                        else:
                            logger.info("No se detectó texto para traducir.")
                    except Exception as ee:
                        logger.error(f"Error en transcripción: {ee}")

                    # Reset buffer
                    audio_buffer.clear()
                    is_speaking = False

                # Limpiar buffer si es demasiado grande (ruido)
                if len(audio_buffer) > buffer_samples - params['SAMPLE_RATE']:
                    logger.warning("Buffer lleno (ruido constante?), reiniciando buffer...")
                    audio_buffer.clear()
                    is_speaking = False

                sd.sleep(50)

        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
            logger.info("Programa detenido por usuario.")
        except Exception as e:
            logger.error(f"\nOcurrió un error: {e}")

if __name__ == "__main__":
    main()