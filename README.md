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

---

## 早見表

| 関数 / メソッド | 説明 |
|---|---|
| `ketsugecam.create(output_idx=0, device_idx=0, region=None)` | カメラを作る |
| `cam.grab()` | 1枚撮る。変化なしなら `None` |
| `cam.start(target_fps)` / `cam.stop()` | 連続キャプチャの開始 / 停止 |
| `cam.get_latest_frame(timeout=1.0, copy=False)` | 最新フレームを取る |
| `cam.release()` | 後片付け |
| `ketsugecam.detector(model_path, conf, iou, max_det)` | 検出器を作る |
| `model.predict(frame)` | 検出して結果のリストを返す |
| `model.draw(img, dets)` | numpy画像に結果を描く |
| `ketsugecam.run(...)` | キャプチャ + 検出 + 表示をまとめて起動 |
| `ketsugecam.monitor_size()` | モニターの `(幅, 高さ)` |
| `ketsugecam.list_monitors()` | 全モニターの座標 |
