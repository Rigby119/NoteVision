import sys
import os
import re
import subprocess
import wave
import numpy as np

# ==============================================================================
# 1. FUNCIONES DE PARSEO Y SÍNTESIS DE AUDIO
# ==============================================================================

def parse_position(pos_str):
    """Convierte la posición L/S del pentagrama a offset diatónico desde L1."""
    is_line = pos_str.startswith('L')
    idx = int(pos_str[1:])
    return (idx - 1) * 2 if is_line else (idx - 1) * 2 + 1

def diatonic_to_freq(diatonic_idx):
    """Convierte índice diatónico (C4 = 0) a frecuencia en Hz."""
    semitones_table = [0, 2, 4, 5, 7, 9, 11]
    octave = diatonic_idx // 7
    note_in_oct = diatonic_idx % 7
    midi_note = 60 + octave * 12 + semitones_table[note_in_oct]
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def parse_sequence(seq_text):
    """Extrae claves, notas, silencios y puntillos."""
    tokens = seq_text.strip().split()
    current_clef = 'G'
    events = []
    
    duration_map = {
        'whole': 4.0,
        'half': 2.0,
        'quarter': 1.0,
        'eighth': 0.5,
        'sixteenth': 0.25,
        'thirty_second': 0.125,
    }
    
    for token in tokens:
        if token.startswith('clef.'):
            clef_type = token.split('.')[1].split('-')[0]
            if clef_type in ['G', 'F']:
                current_clef = clef_type
                
        elif token.startswith('note.') or token.startswith('rest.'):
            parts = token.split('.')
            is_note = (parts[0] == 'note')
            spec = parts[1]
            
            subparts = spec.split('-')
            dur_name = subparts[0]
            pos_str = subparts[1] if len(subparts) > 1 else 'L3'
            
            if dur_name in duration_map:
                beats = duration_map[dur_name]
            elif 'beamed' in dur_name:
                beats = 0.25 if '2' in dur_name else (0.125 if '3' in dur_name else 0.5)
            else:
                beats = 1.0
                
            if is_note:
                step = parse_position(pos_str)
                base_diatonic = 2 if current_clef == 'G' else -10
                freq = diatonic_to_freq(base_diatonic + step)
                events.append({'type': 'note', 'freq': freq, 'duration': beats})
            else:
                events.append({'type': 'rest', 'freq': 0.0, 'duration': beats})
                
        elif token.startswith('dot'):
            if events:
                events[-1]['duration'] *= 1.5
                
    return events

def generate_audio(events, bpm=120, sample_rate=22050):
    """Sintetiza la onda de audio."""
    beat_duration = 60.0 / bpm
    audio_chunks = []
    
    for ev in events:
        duration_sec = ev['duration'] * beat_duration
        num_samples = int(sample_rate * duration_sec)
        
        if num_samples == 0:
            continue
            
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        if ev['type'] == 'rest' or ev['freq'] == 0:
            chunk = np.zeros(num_samples)
        else:
            freq = ev['freq']
            waveform = 0.75 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
            
            env_samples = min(int(sample_rate * 0.01), num_samples // 2)
            envelope = np.ones(num_samples)
            if env_samples > 0:
                envelope[:env_samples] = np.linspace(0, 1, env_samples)
                envelope[-env_samples:] = np.linspace(1, 0, env_samples)
            
            chunk = waveform * envelope
            
        audio_chunks.append(chunk)
        
    if not audio_chunks:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(audio_chunks)

def save_wav(filename, audio_signal, sample_rate=22050):
    """Guarda el array en formato WAV de 16-bit PCM con el módulo nativo wave."""
    audio_int16 = (np.clip(audio_signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit (2 bytes por muestra)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

# ==============================================================================
# 2. EJECUCIÓN Y EXTRACCIÓN DE LA SECUENCIA
# ==============================================================================

def run_predict_script(image_path, predict_script_path="code/predict.py"):
    cmd = [sys.executable, predict_script_path, image_path]
    print(f"Ejecutando: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("\nError al ejecutar predict.py:")
        print(result.stderr)
        sys.exit(result.returncode)
        
    return result.stdout

def extract_sequence(stdout_text):
    match = re.search(r"Predicted sequence\s*\(\d+\s*tokens\):\s*([\s\S]+)", stdout_text)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No se pudo encontrar 'Predicted sequence' en la salida del modelo.")

# ==============================================================================
# 3. MENTRO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python music_predict.py \"<ruta_de_imagen>\" [ruta_a_predict.py]")
        sys.exit(1)
        
    if len(sys.argv) > 2 and sys.argv[-1].endswith(".py"):
        predict_script = sys.argv[-1]
        image_path = " ".join(sys.argv[1:-1])
    else:
        predict_script = "code/predict.py"
        image_path = " ".join(sys.argv[1:])
    
    image_path = image_path.strip("'\"")

    # 1. Obtener la salida de predict.py
    stdout_output = run_predict_script(image_path, predict_script)
    
    # 2. Extraer secuencia
    try:
        sequence_text = extract_sequence(stdout_output)
        print("\n--- Secuencia Extraída ---")
        print(sequence_text)
        print("---------------------------\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Salida recibida de predict.py:")
        print(stdout_output)
        sys.exit(1)
        
    # 3. Sintetizar audio
    events = parse_sequence(sequence_text)
    audio_signal = generate_audio(events, bpm=120)
    
    # 4. Guardar archivo WAV
    output_filename = "output_audio.wav"
    save_wav(output_filename, audio_signal, sample_rate=22050)
    
    abs_path = os.path.abspath(output_filename)
    print(f"✅ Archivo WAV generado y guardado en:\n{abs_path}")