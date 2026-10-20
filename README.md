# ketsugecam 使い方

画面をキャプチャして、そのまま GPU 上の `torch.Tensor` として受け取るライブラリ。
YOLO (TensorRT) 検出もついている。

返ってくるフレームは常に `torch.Tensor`、形は `[高さ, 幅, 3]`、`uint8`、**BGR**、CUDA 上。

## 準備

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA版PyTorch
pip install comtypes
pip install tensorrt opencv-python numpy   # 検出・表示を使うときだけ
```

```python
import ketsugecam
```

---

## 1. 画面を1枚撮る

```python
cam = ketsugecam.create(region=(0, 0, 640, 640))   # (左, 上, 右, 下)
frame = cam.grab()

if frame is not None:        # 画面に変化が無いと None
    print(frame.shape)       # torch.Size([640, 640, 3])

cam.release()
```

`region` を省略すると画面全体。

`with` を使えば `release()` は自動:

```python
with ketsugecam.create() as cam:
    frame = cam.grab()
```

## 2. 連続で撮る

```python
cam = ketsugecam.create(region=(0, 0, 640, 640))
cam.start(target_fps=240)                 # 裏のスレッドで撮り続ける

try:
    while True:
        frame = cam.get_latest_frame()    # 新しいフレームが来るまで待つ (最大1秒)
        if frame is None:
            continue
        ...                               # ここで frame を使う
finally:
    cam.stop()
    cam.release()
```

- `cam.capture_fps` … 実際のFPS
- `cam.mode` … `"cuda_interop"` (GPUだけで完結) か `"cpu_fallback"`
- フレームを後で使うために取っておくなら `get_latest_frame(copy=True)`。
  付けないと次のフレームで中身が上書きされる

## 3. 画面の真ん中を撮る

```python
w, h = ketsugecam.monitor_size()   # 例: (1920, 1080)。DPIスケーリングの影響なし
size = 416
cam = ketsugecam.create(region=((w - size) // 2, (h - size) // 2,
                                (w + size) // 2, (h + size) // 2))
```

## 4. YOLOで検出する

```python
cam   = ketsugecam.create(region=(752, 332, 1168, 748))
model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)
cam.start(target_fps=400)

while True:
    frame = cam.get_latest_frame(copy=True)
    if frame is None:
        continue

    for d in model.predict(frame):
        print(d.name, d.conf, d.xyxy, d.center)
```

ultralytics 風に書くこともできる:

```python
from ketsugecam import YOLO

model_yolo = YOLO("best.engine")                    # スクリプトの隣 -> カレント -> ketsugecam/ 内
dets = model_yolo(frame, conf=0.35, iou=0.45, max_det=5)   # 引数はその呼び出しだけ有効
```

前処理・NMS は中でやるので、`frame` をそのまま渡すだけ。
numpy の BGR 画像を渡してもよい。

検出結果 `d` の中身:

| 属性 | 内容 |
|---|---|
| `xyxy` | 枠 `(x1, y1, x2, y2)` |
| `xywh` | 枠 `(x, y, 幅, 高さ)` |
| `center` | 枠の中心 `(x, y)` |
| `conf` | 信頼度 |
| `cls` / `name` | クラス番号 / クラス名 |

- `"best.engine"` は同梱モデル。自分のモデル (`.engine` / `.onnx`) のパスも渡せる
- クラス名はモデルから自動で読む (`model.names` で確認)
- `model.conf = 0.5` のように、しきい値は途中で変えられる

### 結果を描いて表示する

```python
import cv2

dets = model.predict(frame)
img = frame.cpu().numpy()       # 表示用にCPUへ
model.draw(img, dets)           # 枠・信頼度・中心点を描き込む
cv2.imshow("ketsugecam", img)
cv2.waitKey(1)
```

## 5. 全部おまかせで動かす

キャプチャ → 検出 → 表示を1行で:

```python
ketsugecam.run(
    capture_size=(416, 416),   # 画面中央から撮る大きさ
    model_path="best.engine",
    detect=True,               # False にするとキャプチャだけ
    show_window=True,
    draw_boxes=True,
    show_fps=True,
)
```

コマンドからでも同じ:

```bash
python -m ketsugecam
```

**INSERT** キー (またはプレビュー上で **ESC**) で終了。
設定を変えるときは `__main__.py` 上の定数を書き換える。

## 6. 複数モニター

```python
print(ketsugecam.list_monitors())          # [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]

cam0 = ketsugecam.create(output_idx=0)     # 1枚目のモニター
cam1 = ketsugecam.create(output_idx=1)     # 2枚目のモニター
```

GPU が複数あるときは `device_idx` で選ぶ。

## 7. 遅延を最小にする

`cam.start()` + `get_latest_frame()` は手軽だが、キャプチャスレッドのタイマー周期ぶん
(400fps なら最大 2.5ms) 待ちが入る。遅延を詰めるなら `start()` を呼ばずに、
検出するスレッドで `grab()` を直接呼ぶ:

```python
cam   = ketsugecam.create(region=(752, 332, 1168, 748))   # start() は呼ばない
model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)

while True:
    frame = cam.grab(timeout_ms=100)   # 新フレームが来た瞬間に戻る
    if frame is None:
        continue
    dets = model.predict(frame)        # 次の grab() より前に終わるので copy 不要
```

**このループを別スレッドで回すときは、最初の `grab()` の前に
`torch.cuda.synchronize()` を一度呼ぶ。** `grab()` は「呼ぶ前のコンテキスト」に
戻すが、新しいスレッドには最初どのコンテキストも無いので、D3D11 側のコンテキストが
残ったままになり、`predict()` が `CUDA error: invalid resource handle` で落ちる。

```python
def detection_thread():
    torch.cuda.synchronize()           # torch のコンテキストをこのスレッドでカレントに
    while True:
        frame = cam.grab(timeout_ms=100)
        ...
```

ほかに効くこと:

- **表示用の `.cpu()` を検出ループに入れない。** 表示側が欲しいときだけ
  `frame.clone()` (GPU 上で非同期、数µs) を渡し、`.cpu()` は表示スレッドでやる
- **起動時にウォームアップする。** 最初の `predict()` は CUDA グラフのキャプチャで
  数十 ms かかるので、本番前に数回呼んでおく
- **GIL の切り替え待ちを短くする。** `sys.setswitchinterval(0.0005)`
  (既定 5ms。表示スレッドが GIL を握っている間、検出スレッドが待たされる)
- **Windows の優先度を上げる。** プロセスを HIGH、検出スレッドを TIME_CRITICAL、
  `timeBeginPeriod(1)`

遅延の測り方: `grab()` 直後の `cam._frame_info.LastPresentTime` は、そのフレームが
画面に出た時刻 (QPC)。`QueryPerformanceCounter` との差で「画面に出てから検出結果が
出るまで」が測れる。

コードでは消せない遅延:

- 画面の更新間隔 (144Hz なら 6.9ms、240Hz なら 4.2ms)
- 推論時間そのもの (モデルの大きさ・`imgsz`・FP16 で決まる)

全部入りの例 (検出スレッド + pygame 表示): `新しいフォルダー/main.py`

---

## 早見表

| 関数 / メソッド | 説明 |
|---|---|
| `ketsugecam.create(output_idx=0, device_idx=0, region=None)` | カメラを作る |
| `cam.grab(timeout_ms=1)` | 1枚撮る。変化なしなら `None`。新フレーム到着で即戻る (最小遅延) |
| `cam.start(target_fps)` / `cam.stop()` | 連続キャプチャの開始 / 停止 |
| `cam.get_latest_frame(timeout=1.0, copy=False)` | 最新フレームを取る |
| `cam.release()` | 後片付け |
| `ketsugecam.detector(model_path, conf, iou, max_det)` | 検出器を作る |
| `model.predict(frame)` | 検出して結果のリストを返す |
| `model.draw(img, dets)` | numpy画像に結果を描く |
| `ketsugecam.run(...)` | キャプチャ + 検出 + 表示をまとめて起動 |
| `ketsugecam.monitor_size()` | モニターの `(幅, 高さ)` |
| `ketsugecam.list_monitors()` | 全モニターの座標 |
