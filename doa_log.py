import sounddevice as sd
import numpy as np
import pyroomacoustics as pra
from collections import deque
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# =========================
# Cấu hình
# =========================
DEVICE_INDEX = 2       # Card 2: seeed-4mic-voicecard
CHANNELS = 4           # 4 mic
RATE = 16000           # Sample rate
CHUNK = 1024           # Mỗi lần đọc
BUFFER_SIZE = 5        # Moving average (làm mượt kết quả DOA)

# Vị trí 4 mic (m trong hệ tọa độ 2D) -> cấu hình chuẩn ReSpeaker 4-Mic Array
MIC_POS = np.array([
    [0.04, 0.0],    # Mic 1 (phía phải)
    [0.0, 0.04],    # Mic 2 (phía trên)
    [-0.04, 0.0],   # Mic 3 (phía trái)
    [0.0, -0.04],   # Mic 4 (phía dưới)
]).T  # (2,4)

# Khởi tạo bộ DOA với 360 góc (độ phân giải 1°)
doa = pra.doa.MUSIC(MIC_POS, RATE, nfft=256, c=343, num_src=1)

# Bộ nhớ để làm mượt kết quả
angle_buffer = deque(maxlen=BUFFER_SIZE)

# =========================
# Thiết lập đồ thị la bàn
# =========================
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")   # 0° ở phía Bắc
ax.set_theta_direction(-1)        # Góc tăng theo chiều kim đồng hồ
line, = ax.plot([], [], 'ro-', linewidth=2)  # kim chỉ hướng

def update_compass(angle_deg):
    angle_rad = np.radians(angle_deg)
    line.set_data([angle_rad, angle_rad], [0, 1])  # kim dài từ 0 → 1
    ax.set_title(f"Hướng âm thanh: {angle_deg:.1f}°")
    plt.pause(0.01)

# =========================
# Chuẩn bị file log CSV
# =========================
log_file = open("doa_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "angle_instant", "angle_smoothed"])

# =========================
# Callback xử lý audio
# =========================
def callback(indata, frames, time, status):
    if status:
        print("⚠️ Lỗi ghi âm:", status)

    sig = indata.T  # (channels, samples)

    # Tính STFT
    X = np.fft.rfft(sig, doa.nfft, axis=1)
    doa.locate_sources(X)

    angle = np.degrees(doa.azimuth_recon)[0]
    if angle < 0:
        angle += 360

    angle_buffer.append(angle)
    smooth_angle = np.mean(angle_buffer)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Góc tức thời: {angle:.1f}° | Góc làm mượt: {smooth_angle:.1f}°")

    # Ghi vào CSV
    csv_writer.writerow([ts, f"{angle:.2f}", f"{smooth_angle:.2f}"])
    log_file.flush()

    # Cập nhật biểu đồ
    update_compass(smooth_angle)

# =========================
# Main loop
# =========================
try:
    print("🎤 Bắt đầu ghi âm, vẽ hướng âm thanh và ghi log vào doa_log.csv... (Ctrl+C để dừng)")
    with sd.InputStream(device=DEVICE_INDEX,
                        channels=CHANNELS,
                        samplerate=RATE,
                        blocksize=CHUNK,
                        callback=callback):
        plt.show(block=True)
except KeyboardInterrupt:
    print("\n🛑 Dừng lại.")
finally:
    log_file.close()
