"""
ketsugecam
──────────
BetterCamの無駄を全部削ぎ落とした最速キャプチャライブラリ。

返り値: torch.Tensor(CUDA, uint8, [H, W, 3], BGR)  ← YOLO直投入

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
"""

from .ketsugecam import KetsugeCam

_instance: KetsugeCam | None = None


def create(
    output_idx: int = 0,
    device_idx: int = 0,
    region=None,
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


__all__ = ["create", "KetsugeCam"]