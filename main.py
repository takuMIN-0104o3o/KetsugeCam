"""
ketsugecam を自分のループから呼ぶ例 (検出スレッド + pygame表示)

    python main.py

前処理・TensorRT推論・デコード・NMS・クラス名の読み取りは ketsugecam 内蔵。
こちらで書くのは「設定」と「取得したフレーム/検出結果をどう使うか」だけ。

    pip install pygame
"""

import sys
import threading
import time
from collections import deque
from threading import Event

import cv2
import numpy as np
import pygame

import ketsugecam

# 絵文字つきの print がコンソール(cp932)やパイプで落ちないように
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================= 設定 =================
width_x, height_y = 416, 416
center_screen = (width_x // 2, height_y // 2)

model_path = "best.engine"      # 相対パスは ketsugecam/ 内 -> カレントの順で解決
detection_fps_target = 400
conf_threshold = 0.35
iou_threshold = 0.45
max_detections = 5
display_fps_cap = 60

running = True

# ================= カメラ初期化 =================
# ketsugecam を先に作る: torch の CUDA 初期化より前に CUDA interop の
# コンテキストを用意する必要がある (検出器が初期化する)。
screen_w, screen_h = ketsugecam.monitor_size()
left = (screen_w - width_x) // 2
top = (screen_h - height_y) // 2

camera = ketsugecam.create(
    output_idx=0,
    region=(left, top, left + width_x, top + height_y),
    cuda=True,
)
camera.start(target_fps=detection_fps_target)

# ================= YOLOロード =================
print("🔄 YOLOモデルロード中...")
model = ketsugecam.detector(
    model_path,
    conf=conf_threshold,
    iou=iou_threshold,
    max_det=max_detections,
    agnostic_nms=True,
)
print(f"✅ モデルロード完了 | {model.version.name} | imgsz={model.imgsz} | "
      f"classes={model.names}")
# ウォームアップとCUDAグラフのキャプチャは最初の predict() で済む。

# ================= グローバル =================
_latest_frame_display = None     # CPU numpy (BGR)
_latest_boxes = []
lock = threading.Lock()
frame_ready = Event()            # 新フレーム到着通知

detection_fps = 0
frame_times = deque(maxlen=100)

_stop = Event()                  # 終了
_pause = Event()                 # 一時停止 (SPACE)


# ================= 検出スレッド =================
def detection_thread():
    global _latest_frame_display, _latest_boxes, detection_fps

    frame_count = 0
    last_fps_time = time.time()

    while not _stop.is_set():
        if _pause.is_set():
            time.sleep(0.05)
            continue

        frame_start = time.perf_counter()

        # ketsugecam から GPU テンソルを取得。
        # copy=True: interopの返り値は使い回しバッファのビューなので、
        # 表示用にCPUへ落とすまで中身が書き換わらないようコピーを取る。
        frame_gpu = camera.get_latest_frame(timeout=0.005, copy=True)
        if frame_gpu is None:
            continue

        try:
            # 推論。permute / float / 正規化 / BGR->RGB / NMS は内蔵。
            boxes = model.predict(frame_gpu)

            # 表示用だけCPUへ (推論はGPU上で完結している)
            display_frame = frame_gpu.cpu().numpy()

            with lock:
                _latest_frame_display = display_frame
                _latest_boxes = boxes
            frame_ready.set()

            # FPS / レイテンシ計測
            frame_times.append((time.perf_counter() - frame_start) * 1000)
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                detection_fps = frame_count
                frame_count = 0
                last_fps_time = now
                avg_time = sum(frame_times) / len(frame_times)
                print(f"Detection FPS: {detection_fps} | "
                      f"Latency: {avg_time:.1f}ms")

        except Exception as e:
            print(f"検出エラー: {e}")
            import traceback
            traceback.print_exc()


detect_thread = threading.Thread(target=detection_thread, name="Detection",
                                 daemon=True)
detect_thread.start()
print("✅ 検出スレッド起動")

# ================= Pygame初期化 =================
pygame.init()
screen = pygame.display.set_mode((width_x, height_y))
pygame.display.set_caption("YOLO Detection")
clock = pygame.time.Clock()

_surface_cache = None

# ================= メインループ =================
try:
    while running:
        clock.tick(display_fps_cap)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if _pause.is_set():
                        _pause.clear()
                        print("▶ 検出再開")
                    else:
                        _pause.set()
                        print("⏸ 検出一時停止")

        # 新フレームを待機 (最大16ms)
        if not frame_ready.wait(timeout=0.016):
            continue
        frame_ready.clear()

        with lock:
            if _latest_frame_display is None:
                continue
            frame_bgr = _latest_frame_display
            boxes = _latest_boxes

        # 描画。枠・信頼度・中心点は model.draw() が描く。
        # 自分で描きたいときは d.xyxy / d.conf / d.cls / d.name / d.center を使う。
        model.draw(frame_bgr, boxes, show_name=True)
        cv2.circle(frame_bgr, center_screen, 4, (255, 255, 255), -1)

        # BGR -> RGB して pygame へ (サーフェイスは再利用)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        transposed = frame_rgb.swapaxes(0, 1)
        if _surface_cache is None or _surface_cache.get_size() != (width_x, height_y):
            _surface_cache = pygame.surfarray.make_surface(transposed)
        else:
            pygame.surfarray.blit_array(_surface_cache, transposed)

        screen.blit(_surface_cache, (0, 0))

        pygame.display.set_caption(
            f"YOLO | Detect: {detection_fps} fps | "
            f"Display: {clock.get_fps():.0f} fps | "
            f"Capture: {camera.capture_fps:.0f} fps | "
            f"Detections: {len(boxes)}"
        )
        pygame.display.flip()

finally:
    print("🛑 終了処理中...")
    running = False
    _stop.set()
    detect_thread.join(timeout=3)
    camera.stop()
    camera.release()
    pygame.quit()
    print("✅ 終了")
