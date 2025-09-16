import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import csv

# ====================
# Cấu hình
# ====================
RATE = 16000   # YAMNet yêu cầu 16kHz
DURATION = 2   # Ghi âm 2 giây
CHANNELS = 1   # Dùng 1 kênh (mono)

# Load class map
class_names = []
with open("yamnet_class_map.csv") as f:
    reader = csv.reader(f)
    next(reader)  # bỏ header
    for row in reader:
        class_names.append(row[0])

# Load YAMNet model
interpreter = tflite.Interpreter(model_path="lite-model_yamnet_tflite_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====================
# Ghi âm
# ====================
print("🎙️ Đang ghi âm...")
audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=CHANNELS, dtype="float32")
sd.wait()
print("✅ Ghi âm xong!")

# Reshape audio cho model
audio = np.squeeze(audio)

# YAMNet nhận input 1-D float32
waveform = audio.astype(np.float32)

# Run model
interpreter.set_tensor(input_details[0]['index'], waveform)

predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # (521 classes)

# Top 5 classes
top5 = predictions.argsort()[-5:][::-1]
print("\n🔊 Kết quả phân loại âm thanh:")
for i in top5:
    print(f"- {class_names[i]} ({predictions[i]*100:.2f}%)")
