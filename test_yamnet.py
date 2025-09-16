import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import csv

# =========================
# Cấu hình
# =========================
SAMPLE_RATE = 16000
DURATION = 1.0        # giây
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

# =========================
# Load nhãn YAMNet
# =========================
class_names = []
with open("yamnet_class_map.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # bỏ header
    for row in reader:
        class_names.append(row[2])

# =========================
# Load model TFLite
# =========================
interpreter = tflite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# Ghi âm
# =========================
print("🎙️ Đang ghi âm...")
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
print("✅ Ghi âm xong!")

# Chuyển sang vector 1D
waveform = np.squeeze(recording)

# Chuẩn hóa độ dài đúng 15600 mẫu
if len(waveform) > TARGET_SAMPLES:
    waveform = waveform[:TARGET_SAMPLES]
elif len(waveform) < TARGET_SAMPLES:
    waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))

# Đảm bảo float32
waveform = waveform.astype(np.float32)

# =========================
# Chạy model
# =========================
interpreter.set_tensor(input_details[0]['index'], waveform)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# =========================
# Lấy top-5 kết quả
# =========================
top5_idx = np.argsort(predictions)[::-1][:5]
print("\n🔊 Kết quả phân loại:")
for i in top5_idx:
    print(f"- {class_names[i]} ({predictions[i]:.2f})")
