"""
ketsugecam - 画面キャプチャ + YOLO検出

    python -m ketsugecam

設定はこの6つだけ。残り (クラス名・しきい値・精度・NMS・モデルバージョン判定・
前処理・スレッド構成) は ketsugecam 本体に内蔵。
"""

import os
import sys

if __package__ in (None, ""):
    # `python ketsugecam/__main__.py` のようにファイル直指定で起動した場合、
    # sys.path[0] がこのフォルダになり ketsugecam.py がパッケージを隠すので
    # 親ディレクトリに差し替える (`python -m ketsugecam` では不要)。
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ketsugecam

# ─── 設定 ─────────────────────────────────────────────────────────────────────
CAPTURE_SIZE = (416, 416)      # キャプチャする大きさ (幅, 高さ)。画面中央から
MODEL_PATH   = "best.engine"   # YOLOモデルのパス
DETECT       = True            # YOLO検出を使う / 使わない
SHOW_WINDOW  = True            # 画面表示オン / オフ
DRAW_BOXES   = True            # 検出枠を描く / 描かない
SHOW_FPS     = True            # FPS表示オン / オフ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ketsugecam.run(
        capture_size=CAPTURE_SIZE,
        model_path=MODEL_PATH,
        detect=DETECT,
        show_window=SHOW_WINDOW,
        draw_boxes=DRAW_BOXES,
        show_fps=SHOW_FPS,
    )
