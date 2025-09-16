import numpy as np
import sounddevice as sd
import scipy.signal
import csv
import tflite_runtime.interpreter as tflite

# =============================
# Cấu hình
# =============================
SAMPLE_RATE = 16000  # YAMNet yêu cầu 16kHz
DURATION = 2         # Ghi âm 2 giây
MODEL_PATH = "lite-model_yamnet_tflite_1.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"

# =============================
# Hàm hỗ trợ
# =============================
def record_audio():
    print("🎙️ Đang ghi âm...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("✅ Ghi âm xong!")
    return np.squeeze(audio)

def load_labels(path):
    labels = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # bỏ dòng tiêu đề
        for row in reader:
            labels.append(row[2])  # cột thứ 3: display_name
    return labels

# =============================
# Main
# =============================
if __name__ == "__main__":
    # 1. Load model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Load labels
    class_names = load_labels(CLASS_MAP_PATH)

    # 3. Ghi âm
    waveform = record_audio()

    # 4. Đưa vào model
    # YAMNet input shape: (None,) float32 waveform at 16kHz
    waveform = waveform.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()

    # 5. Lấy output
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # (N, 521 classes)
    mean_scores = np.mean(scores, axis=0)

    # 6. In top-5
    top5 = mean_scores.argsort()[-5:][::-1]
    print("\n🔊 Kết quả phân loại:")
    for i in top5:
        print(f"- {class_names[i]} ({mean_scores[i]:.3f})")
