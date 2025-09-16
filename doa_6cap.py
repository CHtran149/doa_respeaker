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
RATE = 48000            # Tăng sample rate để chính xác hơn
CHUNK = 2048            # Khối mẫu
BUFFER_SIZE = 15        # Buffer dài hơn
MIC_DISTANCE = 0.04     # Khoảng cách giữa các mic (m)
SOUND_SPEED = 343.0     # Tốc độ âm thanh (m/s)

# Bộ nhớ làm mượt
angle_buffer = deque(maxlen=BUFFER_SIZE)

# Ghi log ra CSV
csv_file = open("doa_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Angle (deg)"])

# =========================
# Hàm GCC-PHAT
# =========================
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
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
# La bàn hiển thị
# =========================
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
line, = ax.plot([], [], 'ro-', linewidth=2)

def update_compass(angle_deg):
    angle_rad = float(np.radians(angle_deg))
    if not np.isnan(angle_rad):
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

    # Tính TDOA cho tất cả 6 cặp mic
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    vectors = []

    for (i, j) in pairs:
        tau = gcc_phat(sig[i], sig[j], fs=RATE, max_tau=MIC_DISTANCE/SOUND_SPEED)
        d = tau * SOUND_SPEED
        # Giả sử mic bố trí vuông (x,y), bạn có thể hiệu chỉnh ma trận này theo vị trí mic thực tế
        if (i, j) in [(0,2), (1,3)]:  # cặp ngang
            vectors.append([d, 0])
        else:  # cặp dọc hoặc chéo
            vectors.append([0, d])

    # Least-squares: lấy trung bình vector
    dx = np.mean([v[0] for v in vectors])
    dy = np.mean([v[1] for v in vectors])

    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360

    # Làm mượt bằng median
    angle_buffer.append(angle)
    smooth_angle = np.median(angle_buffer)

    # Xuất kết quả
    print(f"Góc tức thời: {angle:.1f}° | Góc mượt: {smooth_angle:.1f}°")

    # Log CSV
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
