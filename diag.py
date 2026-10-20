"""
ketsugecam/diag.py

検出の揺れ (A) とパイプラインの遅れ (B) を切り分けるための計測。

    python -m ketsugecam.diag --seconds 20 --csv diag.csv
    python -m ketsugecam.diag --seconds 20 --consumer-hz 60    # 表示ループ相当の受け取り

構成は main.py と同じ: 検出スレッドが最新フレームを推論し、利用側スレッドが
最新の結果を受け取る。全フレームの検出を記録し、終了後に集計する。

A (検出位置のノイズ):
    静止した対象を映して計測する。検出をフレーム間で対応づけ (Tracker、予測なし)、
    対象ごとに中心・サイズ・信頼度のばらつきと、フレーム間の飛びを出す。
    視点や対象が動いていると、その動きもばらつきに入る。

B (フレーム・結果の遅れ): 1フレームごとに同じ QPC 時計で
    present → capture_done → infer_start → result_done → consume
    を記録し、p50 / p95 / p99 / max を出す。frame_id で重複処理と飛ばしを数える。
"""

import argparse
import csv
import threading
import time
from collections import defaultdict

from .ketsugecam import QPC_FREQUENCY, qpc_now, qpc_to_ns


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def _std(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def run(seconds=20.0, model_path="best.engine", conf=0.35, iou=0.45, max_det=5,
        consumer_hz=0.0, region_size=416, csv_path=None, min_frames=30):
    import ketsugecam

    w, h = ketsugecam.monitor_size()
    half = region_size // 2
    cam = ketsugecam.create(region=(w // 2 - half, h // 2 - half,
                                    w // 2 + half, h // 2 + half))
    model = ketsugecam.detector(model_path, conf=conf, iou=iou, max_det=max_det,
                                agnostic_nms=True)
    # 予測なしの対応づけ: alpha=1 で位置は観測そのもの、beta=0 で速度は常に 0
    tracker = ketsugecam.Tracker(alpha=1.0, beta=0.0, min_hits=1)

    records = []                  # 1フレーム1件 (dict)
    latest = [None]               # 利用側に渡す最新の結果
    lock = threading.Lock()
    ready = threading.Event()
    stop = threading.Event()
    errors = []

    def detect_loop():
        warm = True
        try:
            while not stop.is_set():
                frame, info = cam.get_latest_frame_info(timeout=0.05, copy=True)
                if frame is None:
                    continue
                if not info.is_new:
                    with lock:
                        stats["repeats"] += 1
                    continue
                dets = model.predict(frame, info)
                tm = model.last_timing
                if tm.warmup or warm:      # CUDA グラフの準備で遅い初回は集計しない
                    warm = False
                    continue
                tracks = tracker.update(dets, info)
                ids = []
                for d in dets:
                    cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
                    tid = next((t.id for t in tracks
                                if t.misses == 0 and abs(t.x - cx) < 1e-6
                                and abs(t.y - cy) < 1e-6 and t.cls == d.cls), 0)
                    ids.append(tid)
                rec = dict(info=info, tm=tm, dets=dets, ids=ids, consume_qpc=0)
                with lock:
                    records.append(rec)
                    latest[0] = rec
                ready.set()
        except Exception as e:     # 計測を止めて原因を表示する
            errors.append(e)
            stop.set()

    stats = {"repeats": 0, "consumes": 0}
    cam.start()
    th = threading.Thread(target=detect_loop, name="diag-detect", daemon=True)
    th.start()
    period = 1.0 / consumer_hz if consumer_hz > 0 else 0.0
    t_end = time.perf_counter() + seconds
    next_t = time.perf_counter()
    while time.perf_counter() < t_end and not stop.is_set():
        if period:
            next_t += period
            d = next_t - time.perf_counter()
            if d > 0:
                time.sleep(d)
        elif not ready.wait(0.05):
            continue
        ready.clear()
        with lock:
            rec = latest[0]
            if rec is not None and rec["consume_qpc"] == 0:
                rec["consume_qpc"] = qpc_now()
                stats["consumes"] += 1
    stop.set()
    th.join(timeout=3)
    cam_stats = cam.get_stats()
    cam.release()
    if errors:
        raise errors[0]
    if not records:
        print("記録できたフレームがありません。")
        return

    if csv_path:
        _write_csv(csv_path, records)
    _report(records, stats, cam_stats, min_frames)


def _write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(("frame_id", "present_qpc", "capture_done_qpc", "infer_start_qpc",
                     "result_done_qpc", "consume_qpc", "gpu_infer_ms", "gpu_decode_ms",
                     "n_dets", "det_idx", "track_id", "cls", "name", "conf",
                     "cx", "cy", "w", "h", "qpc_freq"))
        for r in records:
            info, tm = r["info"], r["tm"]
            head = (info.frame_id, info.present_time_raw, info.capture_done_qpc,
                    tm.start_qpc, tm.done_qpc, r["consume_qpc"] or "",
                    f"{tm.gpu_infer_ms:.4f}", f"{tm.gpu_decode_ms:.4f}", len(r["dets"]))
            if not r["dets"]:
                wr.writerow(head + ("",) * 9 + (QPC_FREQUENCY,))
            for i, (d, tid) in enumerate(zip(r["dets"], r["ids"])):
                x1, y1, x2, y2 = d.xyxy
                wr.writerow(head + (i, tid, d.cls, d.name, f"{d.conf:.4f}",
                                    (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                                    QPC_FREQUENCY))
    print(f"CSV: {path}")


def _report(records, stats, cam_stats, min_frames):
    ms = lambda a, b: qpc_to_ns(b - a) / 1e6

    # ── B: 遅れ ──────────────────────────────────────────────────────────
    ids = [r["info"].frame_id for r in records]
    dup = len(ids) - len(set(ids))
    gaps = sum(b - a - 1 for a, b in zip(ids, ids[1:]) if b > a + 1)
    back = sum(1 for a, b in zip(ids, ids[1:]) if b < a)
    consumed = [r for r in records if r["consume_qpc"]]

    series = {
        "present → capture_done": [ms(r["info"].present_time_raw, r["info"].capture_done_qpc)
                                   for r in records if r["info"].present_time_raw > 0],
        "capture_done → infer_start": [ms(r["info"].capture_done_qpc, r["tm"].start_qpc)
                                       for r in records],
        "infer_start → result_done": [ms(r["tm"].start_qpc, r["tm"].done_qpc)
                                      for r in records],
        "result_done → consume": [ms(r["tm"].done_qpc, r["consume_qpc"]) for r in consumed],
        "present → consume (合計)": [ms(r["info"].present_time_raw, r["consume_qpc"])
                                     for r in consumed if r["info"].present_time_raw > 0],
    }
    span_s = qpc_to_ns(records[-1]["tm"].done_qpc - records[0]["tm"].done_qpc) / 1e9

    print(f"\n=== B: フレーム・結果の遅れ ===")
    print(f"推論したフレーム {len(records)} ({len(records) / span_s:.0f} fps) / "
          f"利用側が受け取った結果 {len(consumed)}")
    print(f"frame_id: 重複 {dup} / 逆順 {back} / 推論せず飛ばした {gaps} / "
          f"同じフレームの再取得 {stats['repeats']}")
    print(f"キャプチャ: DXGI取り逃し {cam_stats.dxgi_skipped} / busy_drops {cam_stats.busy_drops}")
    print(f"{'区間 (ms)':<30}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for name, xs in series.items():
        print(f"{name:<30}{_pct(xs, 50):>8.2f}{_pct(xs, 95):>8.2f}"
              f"{_pct(xs, 99):>8.2f}{max(xs, default=float('nan')):>8.2f}")

    # ── A: 検出のばらつき ───────────────────────────────────────────────
    by_track = defaultdict(list)            # track_id -> [(frame_id, det)]
    n_frames = len(records)
    inferred = sorted(ids)
    for r in records:
        for d, tid in zip(r["dets"], r["ids"]):
            if tid:
                by_track[tid].append((r["info"].frame_id, d))
    tracks = sorted(((tid, obs) for tid, obs in by_track.items() if len(obs) >= min_frames),
                    key=lambda t: -len(t[1]))

    print(f"\n=== A: 検出のばらつき (静止対象を前提) ===")
    print(f"検出のあったフレーム {sum(1 for r in records if r['dets'])}/{n_frames}")
    if not tracks:
        print(f"{min_frames} フレーム以上続いた対象がありません。静止した対象を ROI に入れて計測してください。")
        return
    print(f"{'id':>4} {'name':<10}{'frames':>7}{'途切れ':>7}"
          f"{'σx':>7}{'σy':>7}{'|Δc|p95':>9}{'σw':>7}{'σh':>7}{'conf平均':>9}{'σconf':>7}")
    worst = 0.0
    for tid, obs in tracks[:8]:
        fids = [f for f, _ in obs]
        # 推論したフレームのうち、この対象を見失ったもの (推論しなかった frame_id は数えない)
        import bisect
        lost = (bisect.bisect_right(inferred, fids[-1])
                - bisect.bisect_left(inferred, fids[0])) - len(fids)
        cx = [(d.xyxy[0] + d.xyxy[2]) / 2 for _, d in obs]
        cy = [(d.xyxy[1] + d.xyxy[3]) / 2 for _, d in obs]
        ww = [d.xyxy[2] - d.xyxy[0] for _, d in obs]
        hh = [d.xyxy[3] - d.xyxy[1] for _, d in obs]
        cf = [d.conf for _, d in obs]
        jump = [((cx[i] - cx[i - 1]) ** 2 + (cy[i] - cy[i - 1]) ** 2) ** 0.5
                for i in range(1, len(obs))]
        worst = max(worst, _std(cx), _std(cy))
        print(f"{tid:>4} {obs[0][1].name[:10]:<10}{len(obs):>7}{lost:>7}"
              f"{_std(cx):>7.2f}{_std(cy):>7.2f}{_pct(jump, 95):>9.2f}"
              f"{_std(ww):>7.2f}{_std(hh):>7.2f}{sum(cf) / len(cf):>9.3f}{_std(cf):>7.3f}")
    print("(σ・|Δc| は px。|Δc|p95 = 連続フレーム間の中心移動の p95)")

    # ── 判断の目安 ───────────────────────────────────────────────────────
    p_res = series["infer_start → result_done"]
    p_q = series["capture_done → infer_start"]
    print("\n=== 目安 ===")
    if worst > 1.0:
        print(f"・静止対象で中心の σ が {worst:.2f}px → 検出位置そのものが揺れている (A)")
    else:
        print(f"・中心の σ は {worst:.2f}px 以下 → 検出位置は安定")
    if dup or back:
        print(f"・frame_id の重複 {dup} / 逆順 {back} → 古い結果の混入を調べる (B)")
    else:
        print("・frame_id は重複・逆順なし (古い結果の混入なし)")
    if _pct(p_res, 99) > 3 * _pct(p_res, 50):
        print("・推論時間の裾が長い → GPU の取り合い・スレッド競合を調べる (B)")
    # 推論中に届いたフレームは前の推論が終わるまで待つ: 上限は推論時間程度
    if _pct(p_q, 99) > _pct(p_res, 99) + 1.0:
        print("・capture_done → infer_start が推論時間を超えて待っている → 検出スレッドが詰まっている (B)")
    else:
        print(f"・capture_done → infer_start (p99 {_pct(p_q, 99):.2f}ms) は推論中の待ちの範囲内")


def main(argv=None):
    ap = argparse.ArgumentParser(description="検出の揺れとパイプラインの遅れを計測する")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--model", default="best.engine")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--size", type=int, default=416, help="画面中央の ROI の一辺 (px)")
    ap.add_argument("--consumer-hz", type=float, default=0.0,
                    help="利用側が結果を見る頻度 (0 = 結果が出たらすぐ)")
    ap.add_argument("--min-frames", type=int, default=30,
                    help="A の集計に入れる対象の最低観測フレーム数")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    run(a.seconds, a.model, a.conf, a.iou, a.max_det, a.consumer_hz, a.size,
        a.csv, a.min_frames)


if __name__ == "__main__":
    main()
