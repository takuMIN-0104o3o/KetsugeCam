"""
ketsugecam/latency.py

DXGI の時刻に頼らずにキャプチャ遅延を実測し、LastPresentTime の解釈も検証する。

    python -m ketsugecam.latency                 # 10秒計測
    python -m ketsugecam.latency --seconds 30 --fps 400 --csv latency.csv

仕組み:
    別プロセスのプローブ窓が、描画直前に読んだ QPC (t_draw) を
    64bit + その反転 64bit の白黒マスとして描き続ける。KetsugeCam がその窓を
    キャプチャし、画像から t_draw を復号する。同じ QPC 時計上で

        t_draw ≤ LastPresentTime ≤ acquire_qpc ≤ capture_done_qpc

    が毎フレーム成り立てば、LastPresentTime を QPC として扱う前提が正しい。
    draw → capture_done は DXGI の申告値を使わない「本物の」遅延
    (アプリが描画を始めてから Python が画像を持つまで) になる。

測れないもの:
    ディスプレイのスキャンアウト・液晶応答 (光が出るまで) は画面の外の話なので、
    フォトダイオードやハイスピードカメラが要る。ここで測るのは
    「描画 → デスクトップ合成 → キャプチャ完了 → 利用側が受け取る」まで。
"""

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time

_BITS  = 64
_COLS  = 16                     # 16 x 8 マス = 値 64bit + 反転 64bit
_ROWS  = 8
_CELL  = 16                     # 1マスの辺 (px)
PROBE_W = _COLS * _CELL
PROBE_H = _ROWS * _CELL


def _dpi_aware():
    # DXGI の座標は物理ピクセル。窓の位置もそれに合わせる。
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


# ─── プローブ (別プロセス) ────────────────────────────────────────────────────
def _probe(x: int, y: int):
    """D3D11 レンダラ (フリップモデルのスワップチェーン) で描く。

    GDI で描く通常の pygame.display.flip() は DWM の合成中に窓の中身が
    書き換わりうるので、1枚の中に新旧の値が混ざり (復号失敗)、
    合成開始より後の描画が写り込む (draw > LastPresentTime) ことがある。
    Present 単位で丸ごと入れ替わるレンダラならそれが起きない。
    """
    _dpi_aware()
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_RENDER_DRIVER"] = "direct3d11"
    import pygame
    from pygame._sdl2.video import Window, Renderer

    k32 = ctypes.windll.kernel32
    qpc = ctypes.c_longlong(0)

    pygame.init()
    win = Window("ketsugecam latency probe", size=(PROBE_W, PROBE_H),
                 position=(x, y), borderless=True, always_on_top=True)
    ren = Renderer(win, accelerated=1, vsync=False)

    white, black = (255, 255, 255, 255), (0, 0, 0, 255)
    rects = [pygame.Rect((i % _COLS) * _CELL, (i // _COLS) * _CELL, _CELL, _CELL)
             for i in range(_COLS * _ROWS)]
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
        k32.QueryPerformanceCounter(ctypes.byref(qpc))   # 描画直前の時刻
        v = qpc.value
        for i in range(_BITS):
            bit = (v >> i) & 1
            ren.draw_color = white if bit else black
            ren.fill_rect(rects[i])
            ren.draw_color = black if bit else white
            ren.fill_rect(rects[_BITS + i])
        ren.present()


# ─── 復号 ────────────────────────────────────────────────────────────────────
def _decoder(device):
    import torch
    idx = torch.arange(_COLS * _ROWS, device=device)
    ys = (idx // _COLS) * _CELL + _CELL // 2
    xs = (idx % _COLS) * _CELL + _CELL // 2
    weights = [1 << i for i in range(_BITS)]

    def decode(frame) -> int | None:
        bits = (frame[ys, xs, 1] > 127).tolist()     # G チャンネル
        val, inv = bits[:_BITS], bits[_BITS:]
        if any(a == b for a, b in zip(val, inv)):    # 反転と一致しない → 破損
            return None
        return sum(w for w, b in zip(weights, val) if b)

    return decode


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


# ─── 計測 ────────────────────────────────────────────────────────────────────
def measure(seconds: float = 10.0, target_fps: float | None = None,
            csv_path: str | None = None, x: int | None = None, y: int | None = None,
            output_idx: int = 0):
    _dpi_aware()
    import ketsugecam
    from .ketsugecam import qpc_now, qpc_to_ns

    left, top, right, bottom = ketsugecam.list_monitors()[output_idx]
    if x is None:
        x = left + (right - left - PROBE_W) // 2
    if y is None:
        y = top + (bottom - top - PROBE_H) // 2

    probe = subprocess.Popen(
        [sys.executable, "-m", "ketsugecam.latency", "--probe", str(x), str(y)],
    )
    try:
        time.sleep(1.5)                                   # 窓が出るまで
        cam = ketsugecam.create(
            output_idx=output_idx,
            region=(x - left, y - top, x - left + PROBE_W, y - top + PROBE_H),
        )
        decode = _decoder("cuda" if cam.cuda else "cpu")
        rows, bad_decode, order_ng, stale = [], 0, 0, 0
        last_draw = 0
        cam.start(target_fps=target_fps)
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            with cam.lease_latest_frame(timeout=0.1) as (frame, info):
                consume_qpc = qpc_now()
                if frame is None or not info.is_new:
                    continue
                t_draw = decode(frame)
            if t_draw is None:
                bad_decode += 1
                continue
            if t_draw == last_draw:
                stale += 1          # 画面は更新されたが中身はプローブの前回値
            last_draw = t_draw
            ok = t_draw <= info.present_time_raw <= info.acquire_qpc <= info.capture_done_qpc
            order_ng += not ok
            rows.append((
                info.frame_id, t_draw, info.present_time_raw, info.acquire_qpc,
                info.capture_done_qpc, consume_qpc, info.accumulated_frames, int(ok),
            ))
        stats = cam.get_stats()
        cam.release()
    finally:
        probe.terminate()

    if not rows:
        print("復号できたフレームがありません。プローブ窓が隠れていないか、"
              "DPIスケーリング・位置を確認してください。")
        return

    us = lambda d: qpc_to_ns(d) / 1000
    series = {
        "draw → present   (DXGI申告)": [us(r[2] - r[1]) for r in rows],
        "present → acquire":           [us(r[3] - r[2]) for r in rows],
        "acquire → capture_done":      [us(r[4] - r[3]) for r in rows],
        "draw → capture_done (実測)":  [us(r[4] - r[1]) for r in rows],
        "draw → consume      (実測)":  [us(r[5] - r[1]) for r in rows],
    }

    print(f"\nフレーム {len(rows)} / 復号失敗 {bad_decode} / "
          f"同じ描画値の再取得 {stale} / QPC順序違反 {order_ng}")
    print(f"モード: {'タイマー ' + str(target_fps) + 'fps' if target_fps else '到着待ち'}")
    print(f"QPC 周波数 {ketsugecam.QPC_FREQUENCY} Hz, キャプチャ {stats.capture_fps:.0f} fps, "
          f"DXGI取り逃し {stats.dxgi_skipped}, busy_drops {stats.busy_drops}")
    print(f"{'区間 (µs)':<30}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}{'min':>9}")
    for name, xs in series.items():
        print(f"{name:<30}{_pct(xs, 50):>9.0f}{_pct(xs, 90):>9.0f}"
              f"{_pct(xs, 99):>9.0f}{max(xs):>9.0f}{min(xs):>9.0f}")
    if order_ng == 0:
        print("\n✔ 全フレームで draw ≤ LastPresentTime ≤ acquire ≤ capture_done (同じQPC時計)")
    else:
        print(f"\n✘ {order_ng} フレームで順序違反。LastPresentTime を QPC として扱えない可能性")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(("frame_id", "draw_qpc", "present_qpc", "acquire_qpc",
                        "capture_done_qpc", "consume_qpc", "accumulated_frames",
                        "order_ok", "qpc_freq"))
            for r in rows:
                w.writerow(r + (ketsugecam.QPC_FREQUENCY,))
        print(f"CSV: {csv_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ketsugecam のキャプチャ遅延を実測する")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=None,
                    help="キャプチャの target_fps (省略時は到着待ちモード)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--output", type=int, default=0, help="モニター番号")
    ap.add_argument("--probe", nargs=2, type=int, metavar=("X", "Y"),
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.probe:
        _probe(*a.probe)
    else:
        measure(a.seconds, a.fps, a.csv, output_idx=a.output)


if __name__ == "__main__":
    main()
