"""
ketsugecam/input_latency.py

入力 → 画面反応の遅れ (C) を計測する。検出結果とは無関係に、
「入力を送った時刻」と「画面が変わった時刻」を同じ QPC 時計で記録する。

    python -m ketsugecam.input_latency --trials 40 --dx 100 --csv input.csv
    python -m ketsugecam.input_latency --selftest     # 計測器自体の確認

使い方:
    ゲームを前面にして (視点がマウスで動く状態)、開始までの猶予中に切り替える。
    各試行で SendInput の相対移動 (±dx、交互) を送り、直前のフレームと比べて
    画面中央の ROI が大きく変わった最初のフレームを探す。

測れるもの:
    SendInput → ゲームが入力を読む → 描画 → DWM 合成 (present) → キャプチャ完了
測れないもの:
    マウス本体・USB ポーリング (SendInput は OS より後ろから入れる)、
    ディスプレイのスキャンアウト・液晶応答。光までを測るにはフォトダイオード等が要る。

信頼性:
    変化のしきい値は開始前に入力なしで測ったノイズ (アニメーション等) から決める。
    入力後のフレームを取りこぼした試行は「不確か」として集計から分ける
    (取りこぼしたフレームで先に変わっていたら遅く見積もるため)。
"""

import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import os
import random
import subprocess
import sys
import time

from .ketsugecam import QPC_FREQUENCY, qpc_now, qpc_to_ns

# ─── SendInput ───────────────────────────────────────────────────────────────
_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("mi", _MOUSEINPUT)]


_user32 = ctypes.windll.user32
_user32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wt.UINT


def send_mouse_move(dx: int, dy: int = 0) -> int:
    """相対移動を1回送る。送る直前の QPC を返す。"""
    inp = _INPUT(type=_INPUT_MOUSE,
                 mi=_MOUSEINPUT(dx=dx, dy=dy, dwFlags=_MOUSEEVENTF_MOVE))
    t = qpc_now()
    if _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) != 1:
        raise ctypes.WinError()
    return t


# ─── 画面変化の量 ────────────────────────────────────────────────────────────
def _make_metric(step: int = 4):
    import torch

    def gray(frame):
        return frame[::step, ::step].float().mean(dim=2)

    def diff(a, b) -> float:
        return float((a - b).abs().mean())

    return gray, diff, torch


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


# ─── 計測 ────────────────────────────────────────────────────────────────────
def measure(trials=40, dx=100, size=256, delay_s=3.0, interval_ms=(250, 400),
            timeout_ms=250, threshold=None, csv_path=None, region=None):
    import ketsugecam

    w, h = ketsugecam.monitor_size()
    if region is None:
        half = size // 2
        region = (w // 2 - half, h // 2 - half, w // 2 + half, h // 2 + half)
    cam = ketsugecam.create(region=region)
    gray, diff, torch = _make_metric()
    cam.start()

    def next_frame(timeout=0.1):
        f, info = cam.get_latest_frame_info(timeout=timeout, copy=True)
        if f is None or not info.is_new:
            return None, None
        return gray(f), info

    try:
        print(f"{delay_s:.0f} 秒後に開始。ゲームを前面にしてください。")
        time.sleep(delay_s)

        # ノイズ: 入力なしで連続フレーム間の変化量を測る
        noise, prev = [], None
        t_end = time.perf_counter() + 1.5
        while time.perf_counter() < t_end:
            g, info = next_frame()
            if g is None:
                continue
            if prev is not None:
                noise.append(diff(g, prev))
            prev = g
        floor = _pct(noise, 99) if noise else 0.0
        thr = threshold if threshold is not None else max(3.0 * floor, 2.0)
        print(f"ノイズ: フレーム {len(noise)} / 変化量 p99 {floor:.2f} → しきい値 {thr:.2f}")

        rows, sign = [], 1
        for k in range(trials):
            time.sleep(random.uniform(*interval_ms) / 1000)   # フレーム周期と同期させない
            # 基準: 入力直前の最新フレーム
            base, base_info = None, None
            while base is None:
                base, base_info = next_frame(0.2)
            last_id = base_info.frame_id
            t_in = send_mouse_move(sign * dx)
            sign = -sign

            hit, uncertain, n_seen = None, False, 0
            deadline = t_in + timeout_ms * QPC_FREQUENCY // 1000
            while qpc_now() < deadline:
                g, info = next_frame(0.01)
                if g is None:
                    continue
                if info.frame_id != last_id + 1:
                    uncertain = True       # 間のフレームを見ていない
                last_id = info.frame_id
                if info.present_time_raw and info.present_time_raw < t_in:
                    base = g               # 入力前に更新されたフレーム: 基準を更新
                    continue
                n_seen += 1
                d = diff(g, base)
                if d >= thr:
                    hit = (info, qpc_now(), d)
                    break
            if hit is None:
                rows.append((k, t_in, "", "", "", "", n_seen, int(uncertain)))
                print(f"  #{k:>2}: 反応なし ({timeout_ms}ms 以内)")
                continue
            info, t_seen, d = hit
            rows.append((k, t_in, info.present_time_raw, info.capture_done_qpc, t_seen,
                         f"{d:.2f}", n_seen, int(uncertain)))
        stats = cam.get_stats()
    finally:
        cam.release()

    ok = [r for r in rows if r[2] != "" and not r[7]]
    unc = [r for r in rows if r[2] != "" and r[7]]
    miss = [r for r in rows if r[2] == ""]
    ms = lambda a, b: qpc_to_ns(b - a) / 1e6
    series = {
        "input → present (DXGI)": [ms(r[1], r[2]) for r in ok if r[2]],
        "input → capture_done": [ms(r[1], r[3]) for r in ok],
        "input → 変化を判定": [ms(r[1], r[4]) for r in ok],
    }
    print(f"\n試行 {len(rows)}: 有効 {len(ok)} / 不確か {len(unc)} / 反応なし {len(miss)}"
          f"   (キャプチャ DXGI取り逃し {stats.dxgi_skipped})")
    print(f"{'区間 (ms)':<26}{'p50':>8}{'p95':>8}{'p99':>8}{'min':>8}{'max':>8}")
    for name, xs in series.items():
        if xs:
            print(f"{name:<26}{_pct(xs, 50):>8.2f}{_pct(xs, 95):>8.2f}{_pct(xs, 99):>8.2f}"
                  f"{min(xs):>8.2f}{max(xs):>8.2f}")
    if miss:
        print("反応なしが多い場合: ゲームが前面か、--dx を大きく、--threshold を下げる。")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(("trial", "input_qpc", "present_qpc", "capture_done_qpc",
                         "detected_qpc", "diff", "frames_seen", "uncertain", "qpc_freq"))
            for r in rows:
                wr.writerow(r + (QPC_FREQUENCY,))
        print(f"CSV: {csv_path}")
    return rows


# ─── セルフテスト用プローブ ──────────────────────────────────────────────────
def _probe(x, y, size):
    """マウス移動を受けるたびに白黒が入れ替わる窓。入力 → 描画の経路を持つ
    既知のアプリとして、計測器そのものが反応を拾えるか確かめる用。

    カーソル座標ではなく低レベルフック (WH_MOUSE_LL) で移動そのものを受ける:
    ゲームがカーソルを中央に固定 (ClipCursor) していても、前面でなくても反応する。
    フックは pygame のイベント処理 (メッセージポンプ) の中で呼ばれる。
    """
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_RENDER_DRIVER"] = "direct3d11"
    import pygame
    from pygame._sdl2.video import Window, Renderer

    LRESULT = ctypes.c_ssize_t
    HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wt.WPARAM, wt.LPARAM)
    _user32.CallNextHookEx.argtypes = (wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM)
    _user32.CallNextHookEx.restype = LRESULT
    _user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD)
    _user32.SetWindowsHookExW.restype = wt.HHOOK

    state = [False]

    @HOOKPROC
    def hook(code, wparam, lparam):
        if code == 0 and wparam == 0x0200:          # WM_MOUSEMOVE
            state[0] = not state[0]
        return _user32.CallNextHookEx(None, code, wparam, lparam)

    pygame.init()
    win = Window("ketsugecam input probe", size=(size, size), position=(x, y),
                 borderless=True, always_on_top=True)
    ren = Renderer(win, accelerated=1, vsync=False)
    hhook = _user32.SetWindowsHookExW(14, hook, None, 0)   # WH_MOUSE_LL
    if not hhook:
        raise ctypes.WinError()
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
            v = 255 if state[0] else 0
            ren.draw_color = (v, v, v, 255)
            ren.clear()
            ren.present()
    finally:
        _user32.UnhookWindowsHookEx(hhook)


def selftest(trials=15):
    import ketsugecam
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass
    w, h = ketsugecam.monitor_size()
    size = 256
    x, y = (w - size) // 2, (h - size) // 2
    probe = subprocess.Popen([sys.executable, "-m", "ketsugecam.input_latency",
                              "--probe", str(x), str(y), str(size)])
    try:
        time.sleep(1.5)
        # 移動1回ごとに白黒が反転する (変化量 255)。dx は小さくてよい
        return measure(trials=trials, dx=5, size=size, delay_s=0, threshold=50.0)
    finally:
        probe.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(description="入力 → 画面反応の遅れを計測する")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--dx", type=int, default=100, help="1回の相対移動量 (カウント)")
    ap.add_argument("--size", type=int, default=256, help="画面中央の ROI の一辺 (px)")
    ap.add_argument("--delay", type=float, default=3.0, help="開始までの猶予 (秒)")
    ap.add_argument("--timeout", type=int, default=250, help="反応を待つ上限 (ms)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="変化量のしきい値 (省略時はノイズから自動)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="カーソルに反応するテスト窓で計測器自体を確かめる")
    ap.add_argument("--probe", nargs=3, type=int, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.probe:
        _probe(*a.probe)
    elif a.selftest:
        selftest()
    else:
        measure(a.trials, a.dx, a.size, a.delay, timeout_ms=a.timeout,
                threshold=a.threshold, csv_path=a.csv)


if __name__ == "__main__":
    main()
