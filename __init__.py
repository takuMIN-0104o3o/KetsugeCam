"""
ketsugecam
──────────
BetterCamの無駄を全部削ぎ落とした最速キャプチャライブラリ。

返り値: torch.Tensor(CUDA, uint8, [H, W, 3], BGR)  ← YOLO直投入

インストール:
    pip install -e .          # このリポジトリのルートで
    # 以降どこからでも import できる

使い方:
    import ketsugecam

    # 単発
    cam = ketsugecam.create(region=(0, 0, 640, 640))
    frame = cam.grab()          # torch.Tensor CUDA BGR

    # 連続
    cam.start(target_fps=240)
    while True:
        frame = cam.get_latest_frame()
        if frame is not None:
            results = model(frame)

    # コンテキストマネージャ
    with ketsugecam.create() as cam:
        frame = cam.grab()

    # モニター情報 (region を自前で計算するとき)
    ketsugecam.monitor_size()      # (1920, 1080)
    ketsugecam.list_monitors()     # [(0, 0, 1920, 1080), ...]

YOLO検出つきで丸ごと動かす (キャプチャ→推論→表示を内蔵):

    ketsugecam.run(
        capture_size=(416, 416),   # 画面中央から切り出す大きさ
        model_path="best.engine",  # YOLOモデル
        detect=True,               # 検出する / しない
        show_window=True,          # 画面表示する / しない
        draw_boxes=True,           # 検出枠を描く / 描かない
        show_fps=True,             # FPSを出す / 出さない
    )

    # 追加要件: pip install tensorrt opencv-python numpy

表示もループも自分で書く (検出だけ借りる):

    cam   = ketsugecam.create(region=(752, 332, 1168, 748))
    model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)
    cam.start(target_fps=400)

    while True:
        frame = cam.get_latest_frame(copy=True)   # torch.Tensor CUDA BGR
        if frame is None:
            continue
        for d in model.predict(frame):            # 前処理・NMSは内蔵
            print(d.xyxy, d.conf, d.name, d.center)
"""

from typing import Optional, Tuple

from .ketsugecam import KetsugeCam, list_monitors, monitor_size

__version__ = "0.1.0"


def create(
    output_idx: int = 0,
    device_idx: int = 0,
    region: Optional[Tuple[int, int, int, int]] = None,
    cuda: bool = True,
) -> KetsugeCam:
    """
    KetsugeCamインスタンスを生成して返す。

    Args:
        output_idx: モニター番号 (0=プライマリ)
        device_idx: GPUアダプター番号 (0=メイン)
        region:     キャプチャ範囲 (left, top, right, bottom)
                    None=フルスクリーン
        cuda:       True=torch.Tensor(CUDA)で返す

    Returns:
        KetsugeCam
    """
    return KetsugeCam(
        output_idx=output_idx,
        device_idx=device_idx,
        region=region,
        cuda=cuda,
    )


def detector(
    model_path: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.65,
    max_det: int = 100,
    agnostic_nms: bool = False,
):
    """TensorRT YOLO検出器を作る。自分のループから呼ぶとき用。

        cam   = ketsugecam.create(region=(752, 332, 1168, 748))
        model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)

        frame = cam.grab()                  # torch.Tensor [H, W, 3] uint8 BGR (CUDA)
        for d in model.predict(frame):      # 前処理・デコード・NMSは内蔵
            print(d.xyxy, d.conf, d.name, d.center)

    Args:
        model_path:   YOLOモデル (.engine / .onnx)。相対パスはパッケージ内 ->
                      カレントディレクトリの順に探す。None なら同梱モデル
        conf:         信頼度のしきい値
        iou:          NMSのIoUしきい値
        max_det:      1フレームあたりの最大検出数
        agnostic_nms: True ならクラスをまたいでNMSする

    クラス名はモデルのメタデータから自動で読む。`conf` / `iou` / `max_det` は
    あとから属性として書き換えてもよい。

    Returns:
        ketsugecam.detect.Detector
    """
    # 遅延import: キャプチャだけ使う人に tensorrt / opencv を要求しない。
    from .detect import Detector

    return Detector(model_path=model_path, conf=conf, iou=iou,
                    max_det=max_det, agnostic_nms=agnostic_nms)


def run(
    capture_size: Tuple[int, int] = (416, 416),
    model_path: Optional[str] = None,
    detect: bool = True,
    show_window: bool = True,
    draw_boxes: bool = True,
    show_fps: bool = True,
) -> None:
    """キャプチャ + YOLO検出 + 表示を丸ごと起動する。

    INSERT (またはプレビュー上で ESC) で終了。

    Args:
        capture_size: キャプチャするROIの大きさ (幅, 高さ)。画面中央から切り出す
        model_path:   YOLOモデル (.engine / .onnx)。相対パスはパッケージ内 ->
                      カレントディレクトリの順に探す。None なら同梱モデル
        detect:       False ならキャプチャのみ (TensorRTを読み込まない)
        show_window:  プレビューウィンドウを出すか
        draw_boxes:   検出枠とラベルを描くか
        show_fps:     FPSを表示するか (プレビューへの重ね描きとコンソール表示)

    クラス名・しきい値・精度・NMS・モデルバージョン判定は内部の既定値と
    モデルのメタデータから決まるので、指定する必要はない。

    検出だけ自分のループで使いたい場合は `ketsugecam.create()` と
    `ketsugecam.detect.create_detector()` を直接使う。
    """
    # 遅延import: キャプチャだけ使う人に tensorrt / opencv を要求しない。
    from .app import run as _run

    _run(capture_size=capture_size,
         model_path=model_path,
         detect=detect,
         show_window=show_window,
         draw_boxes=draw_boxes,
         show_fps=show_fps)


__all__ = [
    "create",
    "run",
    "detector",
    "KetsugeCam",
    "list_monitors",
    "monitor_size",
    "__version__",
]
