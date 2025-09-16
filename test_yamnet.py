import numpy as np
import sounddevice as sd
import librosa
import tflite_runtime.interpreter as tflite
import csv, time

# Load YAMNet
interpreter = tflite.Interpreter(model_path="lite-model_yamnet_tflite_1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class map
classes = [line.strip().split(",")[2] for line in open("yamnet_class_map.csv").readlines()[1:]]

# Thu âm
DURATION = 2
RATE = 16000
print("🎙️ Đang ghi âm...")
audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="float32")
sd.wait()
print("✅ Ghi âm xong!")

# Convert thành mono waveform
waveform = np.squeeze(audio)

# ==== Tạo log-mel spectrogram ====
mel_spec = librosa.feature.melspectrogram(
    y=waveform,
    sr=RATE,
    n_fft=512,
    hop_length=160,   # 10ms
    win_length=400,   # 25ms
    n_mels=64,
    fmin=125,
    fmax=7500
)

log_mel = librosa.power_to_db(mel_spec).T  # shape (frames, 64)

# Thêm batch dim
input_data = np.expand_dims(log_mel.astype(np.float32), axis=0)

# Run model
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
preds = interpreter.get_tensor(output_details[0]['index'])[0]

# Top-5 kết quả
top5_idx = preds.argsort()[-5:][::-1]
print("\n🔊 Top-5 âm thanh nhận dạng:")
for i in top5_idx:
    print(f"  {classes[i]} ({preds[i]:.3f})")

# Ghi log CSV
with open("yamnet_log.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), classes[top5_idx[0]], preds[top5_idx[0]]])
