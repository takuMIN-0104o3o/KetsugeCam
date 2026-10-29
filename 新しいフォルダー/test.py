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
  - 表示は 60fps。sleep 明けに want_frame を立てて「その後に来た検出フレーム」を
    描くので、sleep 中に古いフレームを掴んで表示が遅れることはない。
  - HSV は CPU のまま。GPU 版 (OpenCV と完全一致・CUDA グラフ化) も試したが
    416x416 では CPU の方が速く (0.26ms vs 0.35ms)、推論と GPU を取り合うため。
  - GPU→CPU は専用ストリーム + ピン留めメモリ。レガシー既定ストリームで
    同期コピーすると、次の grab() の cuMemcpy2D と直列化されて検出側が待たされる。
  - YOLO.predict の引数退避/復元ラッパを飛ばして Detector.predict を直接呼ぶ。
  - HSV: dilate(3x3, 4回) を 9x9 一発に (結果は同じ)。2値マスクへの threshold は
    無意味なので削除。輪郭は CHAIN_APPROX_SIMPLE (凸包は同じ)。
  - ターゲット決定 (HSV で絞り込み → 重なりをまとめる → 中央に一番近い枠) は
    検出スレッドで毎フレーム。表示 (60fps) を待たない。結果は on_target() に来る。
    GPU→CPU は「枠がかかる行」だけ、HSV は「枠の中」だけ計算。枠が無ければ 0 コスト。

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
from ketsugecam.detect import Detection, Detector

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
display_fps_cap = 60  # 表示だけ 60fps (検出は上限なし)
point_y_ratio = 0.2   # 点の縦位置 (枠の上端=0.0, 中心=0.5, 下端=1.0)

# 色検出設定（事前計算）
color_mode = "yellow"
HSV_RANGES = {
    "red": (np.array([0, 95, 95], dtype=np.uint8), np.array([4, 235, 255], dtype=np.uint8)),
    "purple": (np.array([140, 110, 150], dtype=np.uint8), np.array([150, 195, 255], dtype=np.uint8)),
    "yellow": (np.array([30, 125, 150], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
    "cyan": (np.array([88, 190, 169], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)),
}
lower_hsv, upper_hsv = HSV_RANGES[color_mode]

color_tolerance = 5
# 3x3 矩形を 4 回 dilate するのと 9x9 矩形 1 回は同じ結果
morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

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
model_yolo = YOLO('v/scripts/best.engine', conf=conf_threshold, iou=iou_threshold,
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

# conf/iou/max_det の退避・復元をしない素の predict (設定はロード時のもの)
predict = Detector.predict.__get__(model_yolo)


# ================= ターゲット決定 (検出スレッドで毎フレーム) =================
# キャプチャの元バッファ (H, W, 4) BGRA 連続。grab() が返すのはこれの [..., :3]。
# CUDA interop でないモードでは無いので、そのときは grab() の返り値から取る。
_cap_buf = getattr(camera, "_gpu_buf", None)
# 検出スレッド専用のピン留め受け皿 (表示用とは別)
_det_host = torch.empty((height_y, width_x, 4 if _cap_buf is not None else 3),
                        dtype=torch.uint8, pin_memory=True)
_det_host_np = _det_host.numpy()


def box_has_color(frame_bgra, d):
    """枠の中だけ HSV 判定。一致画素が1つでもあれば True。"""
    h, w = frame_bgra.shape[:2]
    x1, y1, x2, y2 = d.xyxy
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return False
    hsv = cv2.cvtColor(frame_bgra[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)  # BGRA 入力可
    return cv2.countNonZero(cv2.inRange(hsv, lower_hsv, upper_hsv)) > 0


def merge_overlapping_boxes(boxes):
    """重なっている枠をまとめて1つ (外接矩形) にする。連鎖して重なるものも1つに。
    conf / cls / name はまとめた中で一番 conf が高いものを使う。"""
    merged = list(boxes)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        out = []
        for d in merged:
            x1, y1, x2, y2 = d.xyxy
            for i, o in enumerate(out):
                ox1, oy1, ox2, oy2 = o.xyxy
                if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                    best = d if d.conf > o.conf else o
                    out[i] = Detection(
                        xyxy=(min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2)),
                        conf=best.conf, cls=best.cls, name=best.name)
                    changed = True
                    break
            else:
                out.append(d)
        merged = out
    return merged


def aim_point(d):
    """枠の狙い点 (横は中央, 縦は point_y_ratio)。キャプチャ領域内の座標。"""
    x1, y1, x2, y2 = d.xyxy
    return (x1 + x2) // 2, y1 + int((y2 - y1) * point_y_ratio)


def pick_target(boxes, frame):
    """HSV 絞り込み → 重なりをまとめる → 中央に一番近い枠。
    戻り値: (表示する枠のリスト, ターゲットの index か -1)"""
    if not boxes:
        return [], -1
    # 枠がかかる行だけ GPU -> CPU (行単位なので連続コピー 1 回)
    ys1 = max(min(d.xyxy[1] for d in boxes), 0)
    ys2 = min(max(d.xyxy[3] for d in boxes), height_y)
    if ys2 <= ys1:
        return [], -1
    src = _cap_buf if _cap_buf is not None else frame
    _det_host[ys1:ys2].copy_(src[ys1:ys2])

    boxes = [d for d in boxes if box_has_color(_det_host_np, d)]
    if not boxes:
        return [], -1
    boxes = merge_overlapping_boxes(boxes)

    cx, cy = width_x // 2, height_y // 2
    best, best_d2 = -1, None
    for i, d in enumerate(boxes):
        px, py = aim_point(d)
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = i, d2
    return boxes, best


def on_target(target, point, t_dxgi, t_present):
    """毎検出フレーム、検出スレッドから呼ばれる (表示を待たない)。
    target: Detection か None / point: 狙い点 (キャプチャ領域内の座標) か None。
    画面座標が欲しいときは (left + point[0], top + point[1])。

    座標はすべて「t_dxgi に DXGI から受け取ったフレーム」のもの。
      t_dxgi   : そのフレームを DXGI から受け取った瞬間の QPC
      t_present: そのフレームが present された QPC (LastPresentTime)
      経過時間 ms = (qpc_now() - t_dxgi) * 1000 / _qpc_freq
    ここは検出ループの中なので重い処理は書かないこと。"""
    pass

# ================= グローバル =================
_latest_frame_gpu = None         # 表示用スナップショット (GPU, 表示側が要求したときだけ)
_latest_frame_evt = None         # そのクローン完了を示す CUDA イベント
_latest_boxes = []               # 絞り込み・まとめ済みの枠
_latest_target = -1              # その中のターゲット index (-1 = なし)
lock = threading.Lock()
frame_ready = Event()            # 新しい表示用フレームあり
want_frame = Event()             # 表示側が次のフレームを欲しがっている

detection_fps = 0
proc_ms = deque(maxlen=200)      # DXGI から受け取ってからターゲットが決まるまで
age_ms = deque(maxlen=200)       # present されてから DXGI に届くまで

_stop = Event()                  # 終了
_pause = Event()                 # 一時停止 (SPACE)


# ================= 検出スレッド =================
def detection_thread():
    global _latest_frame_gpu, _latest_frame_evt, _latest_boxes, _latest_target
    global detection_fps

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
        t_dxgi = camera.acquire_qpc      # DXGI から受け取った瞬間 (コピー前)
        t_present = camera._frame_info.LastPresentTime

        try:
            # 推論。permute / float / 正規化 / BGR->RGB / NMS は内蔵。
            boxes = predict(frame_gpu)

            # ターゲット決定もこのフレームのうちに (次の grab() より前なので
            # キャプチャバッファはまだ上書きされていない)
            boxes, target_idx = pick_target(boxes, frame_gpu)
            if target_idx >= 0:
                target = boxes[target_idx]
                on_target(target, aim_point(target), t_dxgi, t_present)
            else:
                on_target(None, None, t_dxgi, t_present)
            t_done = qpc_now()
            errors_in_row = 0

            # 表示側が要求しているときだけ GPU 上でクローン (非同期)。
            # クローンは既定ストリームのまま (次の grab() の cuMemcpy2D より
            # 前に終わる順序保証のため)。完了イベントを表示側に渡す。
            snap = evt = None
            if want_frame.is_set():
                snap = frame_gpu.clone()
                evt = torch.cuda.Event()
                evt.record()

            with lock:
                _latest_boxes = boxes
                _latest_target = target_idx
                if snap is not None:
                    _latest_frame_gpu = snap
                    _latest_frame_evt = evt
                    want_frame.clear()
            if snap is not None:
                frame_ready.set()

            # ---- ここから下は計測のみ ----
            proc_ms.append((t_done - t_dxgi) * 1000.0 / _qpc_freq)
            if t_present:
                age_ms.append((t_dxgi - t_present) * 1000.0 / _qpc_freq)
            frame_count += 1
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                detection_fps = round(frame_count / (now - last_fps_time))
                frame_count = 0
                last_fps_time = now
                p = sum(proc_ms) / len(proc_ms)
                a = sum(age_ms) / len(age_ms) if age_ms else float("nan")
                print(f"Detection FPS: {detection_fps} | "
                      f"present→DXGI: {a:.2f}ms | DXGI→ターゲット: {p:.2f}ms")

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

# 表示用 GPU→CPU 転送: 専用 (non-blocking) ストリーム + ピン留め受け皿
_copy_stream = torch.cuda.Stream()
_host_frame = torch.empty((height_y, width_x, 3), dtype=torch.uint8, pin_memory=True)
_host_np = _host_frame.numpy()          # 同じメモリの numpy ビュー
_draw_buf = np.empty_like(_host_np)     # 描画用 (ピン留め側は次の転送で上書きされる)


# ================= HSV輪郭 (表示用) =================
def detect_hsv_hulls(frame, lower_hsv, upper_hsv, morph_element):
    """HSV色検知の輪郭 (凸包) を返す。表示専用 (ターゲット判定は検出スレッド)。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.dilate(cv2.inRange(hsv, lower_hsv, upper_hsv), morph_element)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.convexHull(c) for c in contours] if contours else []


# ================= 描画ヘルパ =================
def hull_touches_boxes(hull, boxes):
    """凸包の外接矩形がどれかの YOLO 枠と重なっていれば True。"""
    hx, hy, hw, hh = cv2.boundingRect(hull)
    for d in boxes:
        x1, y1, x2, y2 = d.xyxy
        if hx < x2 and hx + hw > x1 and hy < y2 and hy + hh > y1:
            return True
    return False


def draw_boxes(frame_bgr, boxes, target_idx):
    """枠 (太さ1) と点 (青, 中心より少し上) を描く。ラベル・信頼度は描かない。
    ターゲット (画面中央に一番近い枠) だけ黄色、それ以外は緑。"""
    for i, d in enumerate(boxes):
        x1, y1, x2, y2 = d.xyxy
        color = (0, 255, 255) if i == target_idx else (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)
        cv2.circle(frame_bgr, aim_point(d), 2, (255, 0, 0), -1)


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

        # sleep 明けに要求 → 次の (最新の) 検出フレームを待つ
        want_frame.set()
        if not frame_ready.wait(timeout=0.016):
            continue
        frame_ready.clear()

        with lock:
            snap = _latest_frame_gpu
            evt = _latest_frame_evt
            boxes = _latest_boxes
            target_idx = _latest_target
            _latest_frame_gpu = None
            _latest_frame_evt = None
        if snap is None:
            continue

        # GPU -> CPU は表示スレッドの専用ストリームで (検出経路を止めない)
        with torch.cuda.stream(_copy_stream):
            _copy_stream.wait_event(evt)
            _host_frame.copy_(snap, non_blocking=True)
        _copy_stream.synchronize()
        del snap
        frame_bgr = _draw_buf
        np.copyto(frame_bgr, _host_np)

        # boxes は検出スレッドで絞り込み・まとめ済み。ここは描くだけ。
        # HSV の輪郭は YOLO 枠にかかっているものだけ、切らずにそのまま描く
        hulls = []
        if boxes:
            hulls = detect_hsv_hulls(frame_bgr, lower_hsv, upper_hsv, morph_element)
            hulls = [h for h in hulls if hull_touches_boxes(h, boxes)]
            if hulls:
                cv2.drawContours(frame_bgr, hulls, -1, (255, 0, 255), 1)

        # 枠 (太さ1) と点。ターゲットは黄色
        draw_boxes(frame_bgr, boxes, target_idx)

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
                f"Detections: {len(boxes)} | {color_mode}: {len(hulls)}"
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
