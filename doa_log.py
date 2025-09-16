import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
import time

# =========================
# Cấu hình
# =========================
DEVICE_INDEX = 2        # Card 2: seeed-4mic-voicecard
CHANNELS = 4            # 4 mic
RATE = 16000            # Sample rate
CHUNK = 1024            # Số mẫu mỗi lần đọc
BUFFER_SIZE = 5         # Moving average
MIC_DISTANCE = 0.04     # Khoảng cách giữa các mic (m)
SOUND_SPEED = 343.0     # Tốc độ âm thanh (m/s)

# Bộ nhớ để làm mượt kết quả
angle_buffer = deque(maxlen=BUFFER_SIZE)

# Ghi log ra CSV
csv_file = open("doa_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Angle (deg)"])

# =========================
# Hàm GCC-PHAT
# =========================
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """Tính độ trễ giữa 2 tín hiệu bằng GCC-PHAT"""
    n = sig.shape[0] + refsig.shape[0]

    # FFT
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-15), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau

# =========================
# Thiết lập đồ thị la bàn
# =========================
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")   # 0° = Bắc
ax.set_theta_direction(-1)        # chiều kim đồng hồ
line, = ax.plot([], [], 'ro-', linewidth=2)

def update_compass(angle_deg):
    angle_rad = np.radians(angle_deg)
    line.set_data([angle_rad, angle_rad], [0, 1])
    ax.set_title(f"Hướng âm thanh: {angle_deg:.1f}°")
    plt.pause(0.01)

# =========================
# Callback xử lý audio
# =========================
def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Lỗi ghi âm:", status)

    sig = indata.T  # (channels, samples)

    # Lấy cặp mic ngang (Mic1–Mic3), dọc (Mic2–Mic4)
    tau_x = gcc_phat(sig[0], sig[2], fs=RATE, max_tau=MIC_DISTANCE/SOUND_SPEED)
    tau_y = gcc_phat(sig[1], sig[3], fs=RATE, max_tau=MIC_DISTANCE/SOUND_SPEED)

    # Chuyển đổi TDOA sang góc
    dx = tau_x * SOUND_SPEED
    dy = tau_y * SOUND_SPEED

    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360

    # Làm mượt
    angle_buffer.append(angle)
    smooth_angle = np.mean(angle_buffer)

    # Xuất kết quả
    print(f"Góc tức thời: {angle:.1f}° | Góc làm mượt: {smooth_angle:.1f}°")

    # Ghi log CSV
    csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{smooth_angle:.1f}"])
    csv_file.flush()

    # Vẽ la bàn
    update_compass(smooth_angle)

# =========================
# Chạy
# =========================
print("🎤 Bắt đầu ghi âm và tính DOA (Ctrl+C để dừng)...")
with sd.InputStream(device=DEVICE_INDEX,
                    channels=CHANNELS,
                    samplerate=RATE,
                    blocksize=CHUNK,
                    callback=callback):
    plt.show(block=True)

csv_file.close()
