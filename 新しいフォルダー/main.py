"""
ketsugecam を自分のループから呼ぶ例 (低遅延版: 検出スレッド + pygame表示)

    python main.py

前処理・TensorRT推論・デコード・NMS・クラス名の読み取りは ketsugecam 内蔵。
こちらで書くのは「設定」と「取得したフレーム/検出結果をどう使うか」だけ。

    pip install pygame-ce

遅延を削ったところ:
  - キャプチャスレッド (camera.start) を使わない。検出スレッドで grab() を直接
    呼ぶと AcquireNextFrame が新フレーム到着の瞬間に戻るので、タイマー周期
    (400fps なら最大 2.5ms) の待ちと、スレッド間の受け渡しが消える。
  - grab() → predict() が同じスレッドで直列なので copy=True のクローンも不要。
  - 表示用の GPU→CPU 転送は検出経路から外す。表示側が欲しいときだけ GPU 上で
    クローン (非同期・数µs) し、CPU への転送は表示スレッドでやる。
  - GIL の切り替え間隔を 5ms → 0.5ms。表示スレッドが GIL を握っていても
    検出スレッドがすぐ戻れる。
  - プロセス優先度 HIGH、検出スレッド TIME_CRITICAL、タイマー分解能 1ms。
  - 表示は cvtColor / swapaxes をやめて BGR のまま pygame に渡す。

残る遅延 (アプリ側では消せない):
  - DWM の合成 → DXGI に新フレームが来るまで (モニターのリフレッシュ周期に依存)
  - TensorRT の推論時間そのもの (モデルの大きさ / imgsz / FP16 で決まる)
"""

import ctypes
import sys
import threading
import time
from collections import deque
from threading import Event

import cv2
import numpy as np
import pygame
import torch

import ketsugecam
from ketsugecam import YOLO

# 絵文字つきの print がコンソール(cp932)やパイプで落ちないように
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================= 設定 =================
width_x, height_y = 416, 416

conf_threshold = 0.35
iou_threshold = 0.45
max_detections = 5
display_fps_cap = 60

running = True

# ================= OS 側の遅延対策 =================
_k32 = ctypes.windll.kernel32
_k32.GetCurrentThread.restype = ctypes.c_void_p
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
_k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

_HIGH_PRIORITY_CLASS = 0x80
_THREAD_PRIORITY_TIME_CRITICAL = 15

_k32.SetPriorityClass(_k32.GetCurrentProcess(), _HIGH_PRIORITY_CLASS)
ctypes.windll.winmm.timeBeginPeriod(1)   # sleep / clock.tick の分解能を 1ms に
sys.setswitchinterval(0.0005)            # GIL 切り替え 5ms -> 0.5ms

_qpc_freq = ctypes.c_int64()
_k32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))
_qpc_freq = _qpc_freq.value
_qpc_buf = ctypes.c_int64()


def qpc_now() -> int:
    """QPC (DXGI の LastPresentTime と同じ時計)。"""
    _k32.QueryPerformanceCounter(ctypes.byref(_qpc_buf))
    return _qpc_buf.value


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
# camera.start() は呼ばない: 検出スレッドで grab() を直接呼ぶ。
print(f"キャプチャモード: {camera.mode}")

# ================= YOLOロード =================
# 相対パスは main.py のフォルダ -> カレント -> ketsugecam/ 内 の順で探す
print("🔄 YOLOモデルロード中...")
model_yolo = YOLO('best.engine', conf=conf_threshold, iou=iou_threshold,
                  max_det=max_detections, agnostic_nms=True)
print(f"✅ モデルロード完了 | {model_yolo.model_path} | {model_yolo.version.name} | "
      f"imgsz={model_yolo.imgsz} | classes={model_yolo.names}")

# ウォームアップ (CUDAグラフのキャプチャ) を起動時に済ませておく。
# 最初の本番フレームが数十ms 遅れるのを防ぐ。
_warm = None
for _ in range(200):
    _warm = camera.grab(timeout_ms=50)
    if _warm is not None:
        break
if _warm is None:
    _warm = torch.zeros((height_y, width_x, 3), dtype=torch.uint8, device="cuda")
for _ in range(3):
    model_yolo(_warm)
del _warm
print("✅ ウォームアップ完了")

# ================= グローバル =================
_latest_frame_gpu = None         # 表示用スナップショット (GPU, 表示側が要求したときだけ)
_latest_boxes = []
lock = threading.Lock()
frame_ready = Event()            # 新しい表示用フレームあり
want_frame = Event()             # 表示側が次のフレームを欲しがっている

detection_fps = 0
proc_ms = deque(maxlen=200)      # grab が戻ってから検出結果が出るまで
age_ms = deque(maxlen=200)       # 画面に present されてから検出結果が出るまで

_stop = Event()                  # 終了
_pause = Event()                 # 一時停止 (SPACE)


# ================= 検出スレッド =================
def detection_thread():
    global _latest_frame_gpu, _latest_boxes, detection_fps

    _k32.SetThreadPriority(_k32.GetCurrentThread(), _THREAD_PRIORITY_TIME_CRITICAL)

    # 最初の grab() より前に torch の CUDA コンテキストをこのスレッドでカレントにする。
    # grab() は D3D11 側のコンテキストに切り替えたあと「呼ぶ前のコンテキスト」に
    # 戻すが、新しいスレッドには最初どのコンテキストも無い。そのままだと
    # D3D11 側がカレントに残り、predict() が torch のストリームを使えず
    # "invalid resource handle" になる。
    torch.cuda.set_device(0)
    torch.cuda.synchronize()

    errors_in_row = 0
    frame_count = 0
    last_fps_time = time.perf_counter()

    while not _stop.is_set():
        if _pause.is_set():
            time.sleep(0.05)
            continue

        # 新フレームが来た瞬間に戻る (timeout は終了チェックのためだけ)。
        # 返り値は使い回しバッファのビューだが、次の grab() は predict() の
        # 後なので copy 不要。
        frame_gpu = camera.grab(timeout_ms=100)
        if frame_gpu is None:
            continue
        t_grab = qpc_now()
        t_present = camera._frame_info.LastPresentTime

        try:
            # 推論。permute / float / 正規化 / BGR->RGB / NMS は内蔵。
            boxes = model_yolo(frame_gpu)
            t_done = qpc_now()
            errors_in_row = 0

            # 表示側が要求しているときだけ GPU 上でクローン (非同期)。
            # CPU への転送は表示スレッドでやるので、ここでは待たない。
            snap = frame_gpu.clone() if want_frame.is_set() else None

            with lock:
                _latest_boxes = boxes
                if snap is not None:
                    _latest_frame_gpu = snap
                    want_frame.clear()
            if snap is not None:
                frame_ready.set()

            # ---- ここから下は計測のみ ----
            proc_ms.append((t_done - t_grab) * 1000.0 / _qpc_freq)
            if t_present:
                age_ms.append((t_done - t_present) * 1000.0 / _qpc_freq)
            frame_count += 1
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                detection_fps = round(frame_count / (now - last_fps_time))
                frame_count = 0
                last_fps_time = now
                p = sum(proc_ms) / len(proc_ms)
                a = sum(age_ms) / len(age_ms) if age_ms else float("nan")
                print(f"Detection FPS: {detection_fps} | "
                      f"推論: {p:.2f}ms | present→結果: {a:.2f}ms")

        except Exception as e:
            print(f"検出エラー: {e}")
            import traceback
            traceback.print_exc()
            # CUDA エラーは一度出ると続くことが多い。ログを埋めないよう止める
            errors_in_row += 1
            if errors_in_row >= 10:
                print("🛑 検出エラーが10回続いたので検出スレッドを止めます")
                return


detect_thread = threading.Thread(target=detection_thread, name="Detection",
                                 daemon=True)
detect_thread.start()
print("✅ 検出スレッド起動")

# ================= Pygame初期化 =================
pygame.init()
screen = pygame.display.set_mode((width_x, height_y))
pygame.display.set_caption("YOLO Detection")
clock = pygame.time.Clock()


# ================= 描画ヘルパ =================
def draw_boxes(frame_bgr, boxes):
    """枠 (太さ1) とクラス名だけを描く。信頼度・中心点は描かない。"""
    if not boxes:
        return
    for d in boxes:
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        name = d.name
        if name:
            cv2.putText(
                frame_bgr, name, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
            )


# ================= メインループ =================
last_caption = 0.0
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

        # 次の検出フレームをもらう (検出側は要求があるときだけクローンする)
        want_frame.set()
        if not frame_ready.wait(timeout=0.016):
            continue
        frame_ready.clear()

        with lock:
            snap = _latest_frame_gpu
            boxes = _latest_boxes
            _latest_frame_gpu = None
        if snap is None:
            continue

        # GPU -> CPU は表示スレッドで (検出経路を止めない)
        frame_bgr = np.ascontiguousarray(snap.cpu().numpy())

        # 枠 (太さ1) とクラス名だけ描く
        draw_boxes(frame_bgr, boxes)

        # BGR のまま pygame へ (cvtColor / 転置なし)
        surface = pygame.image.frombuffer(frame_bgr, (width_x, height_y), "BGR")
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # キャプション更新は重い (ウィンドウメッセージ) ので 4Hz に間引く
        now = time.perf_counter()
        if now - last_caption >= 0.25:
            last_caption = now
            pygame.display.set_caption(
                f"YOLO | Detect: {detection_fps} fps | "
                f"Display: {clock.get_fps():.0f} fps | "
                f"Detections: {len(boxes)}"
            )

finally:
    print("🛑 終了処理中...")
    running = False
    _stop.set()
    detect_thread.join(timeout=3)
    camera.release()
    pygame.quit()
    ctypes.windll.winmm.timeEndPeriod(1)
    print("✅ 終了")
