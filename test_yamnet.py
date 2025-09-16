import sounddevice as sd
import numpy as np
import tflite_runtime.interpreter as tflite
import csv

# =====================
# Cấu hình
# =====================
MODEL_PATH = "lite-model_yamnet_tflite_1.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"
SAMPLE_RATE = 16000
TARGET_SAMPLES = 15600   # input chuẩn của YAMNet (~0.975s)

# =====================
# Load class map
# =====================
class_names = []
with open(CLASS_MAP_PATH, "r") as f:
    reader = csv.reader(f)
    next(reader)  # bỏ header
    for row in reader:
        class_names.append(row[2])  # cột Display Name

# =====================
# Load model
# =====================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================
# Ghi âm 1 giây
# =====================
print("🎙️ Đang ghi âm...")
waveform = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
sd.wait()
print("✅ Ghi âm xong!")

waveform = np.squeeze(waveform)  # (16000,)

# =====================
# Chuẩn hóa thành 15600 mẫu
# =====================
if len(waveform) > TARGET_SAMPLES:
    waveform = waveform[:TARGET_SAMPLES]
elif len(waveform) < TARGET_SAMPLES:
    waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))

# Thêm batch dimension
waveform = np.expand_dims(waveform, axis=0)  # (1, 15600)

# =====================
# Chạy model
# =====================
interpreter.set_tensor(input_details[0]['index'], waveform)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # (521,)

# =====================
# Lấy nhãn top-1
# =====================
top_index = np.argmax(predictions)
top_score = predictions[top_index]
top_class = class_names[top_index]

print(f"🔊 Âm thanh dự đoán: {top_class} (score={top_score:.2f})")
