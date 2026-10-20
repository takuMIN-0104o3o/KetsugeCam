"""
ketsugecam/predict.py

検出結果をフレームの時刻で追跡し、遅延ぶん先の位置を予測する。

    tracker = ketsugecam.Tracker()
    frame, info = cam.get_latest_frame_info(copy=True)
    if frame is not None and info.is_new:
        tracker.update(model.predict(frame), info)
    for tr, (x, y) in tracker.predict_all(extra_ms=0.0):
        ...   # (x, y) は「今 + extra_ms」時点の推定中心 (ROI内ピクセル)

時刻の扱い:
    観測時刻   = そのフレームの LastPresentTime (QPC)。画面がその状態だった時刻に
                 一番近い。取れなければ capture_done_qpc。
    予測先時刻 = qpc_now() + extra_ms。extra_ms はこの後ろに続く遅延
                 (表示・入力など、ここでは測れないもの) を利用側が足す。
    よって補正量 = 速度 × (今 − present + extra_ms) で、キャプチャ・推論の遅延は
    毎フレームの実測値がそのまま入る。

注意:
    座標は画面 (ROI) 上なので、視点そのものが動くとその動きも速度に入る。
    等速モデルなので、急な方向転換・停止の直後は予測が行き過ぎる。
    予測距離は max_lead_ms で頭打ちにしてある。
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .ketsugecam import FrameInfo, QPC_FREQUENCY, qpc_now


@dataclass
class Track:
    """1対象の追跡状態 (α-βフィルタ、等速モデル)。"""
    id: int
    cls: int
    x: float                    # 中心 (px)
    y: float
    vx: float = 0.0             # 速度 (px/s)
    vy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    t_qpc: int = 0              # 最後に観測したフレームの時刻 (QPC)
    hits: int = 1               # 観測できた回数
    misses: int = 0             # 連続で見失った回数
    name: str = ""
    conf: float = 0.0           # 最後の検出の信頼度
    raw_x: float = 0.0          # 最後の検出の中心そのもの (平滑化前)
    raw_y: float = 0.0
    # 直近の観測そのもの [(t_qpc, cx, cy, w, h)]。静止判定と静止中の平均に使う
    hist: deque = field(default_factory=lambda: deque(maxlen=64), repr=False)

    def predict(self, t_qpc: int, max_lead_ms: float = 50.0) -> Tuple[float, float]:
        """t_qpc 時点の推定中心。"""
        dt = (t_qpc - self.t_qpc) / QPC_FREQUENCY
        dt = max(0.0, min(dt, max_lead_ms * 1e-3))
        return self.x + self.vx * dt, self.y + self.vy * dt

    def lead(self, t_qpc: int, max_lead_ms: float = 50.0) -> Tuple[float, float]:
        """補正量 = 推定中心 − 最後の観測中心 (px)。"""
        px, py = self.predict(t_qpc, max_lead_ms)
        return px - self.x, py - self.y

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


class Tracker:
    """検出をフレーム間で対応づけ、各対象の位置と速度を推定する。

    Args:
        alpha:        位置の追従率 (0..1)。大きいほど観測に素直、小さいほど滑らか
        beta:         速度の追従率 (0..1)。大きいほど加減速に早く追従、ノイズも増える
        gate_px:      予測位置からこの距離以内の検出だけを同じ対象とみなす
        max_misses:   この回数連続で見失ったら追跡をやめる
        min_hits:     この回数観測するまでは predict_all() に出さない (誤検出対策)
        max_lead_ms:  予測する時間の上限
        size_alpha:   枠の幅・高さの追従率 (0..1)。小さいほど大きさの揺れが減る
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.1,
                 gate_px: float = 60.0, max_misses: int = 5,
                 min_hits: int = 3, max_lead_ms: float = 50.0,
                 size_alpha: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gate_px = gate_px
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.max_lead_ms = max_lead_ms
        self.size_alpha = size_alpha
        self.tracks: List[Track] = []
        self._next_id = 1

    @staticmethod
    def frame_time(info: FrameInfo) -> int:
        return info.present_time_raw if info.present_time_raw > 0 else info.capture_done_qpc

    def update(self, detections: Sequence, info: FrameInfo) -> List[Track]:
        """1フレーム分の検出 (ketsugecam.detect.Detection) で追跡を更新する。

        info.is_new が False (同じフレームの再取得) なら何もしない:
        同じ観測を2回入れると速度が 0 に引っ張られる。
        """
        if not info.is_new:
            return self.tracks
        t = self.frame_time(info)

        # 予測位置との距離で貪欲に対応づける (近い組から確定)
        preds = [tr.predict(t, self.max_lead_ms) for tr in self.tracks]
        pairs = []
        for ti, (px, py) in enumerate(preds):
            for di, d in enumerate(detections):
                cx, cy = _center(d)
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist <= self.gate_px and d.cls == self.tracks[ti].cls:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._correct(self.tracks[ti], preds[ti], detections[di], t)

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        for di, d in enumerate(detections):
            if di not in used_d:
                cx, cy = _center(d)
                x1, y1, x2, y2 = d.xyxy
                tr = Track(
                    id=self._next_id, cls=d.cls, x=cx, y=cy,
                    w=x2 - x1, h=y2 - y1, t_qpc=t, name=getattr(d, "name", ""),
                    conf=getattr(d, "conf", 0.0), raw_x=cx, raw_y=cy,
                )
                tr.hist.append((t, cx, cy, x2 - x1, y2 - y1))
                self.tracks.append(tr)
                self._next_id += 1
        return self.tracks

    def _correct(self, tr: Track, pred: Tuple[float, float], d, t: int):
        dt = (t - tr.t_qpc) / QPC_FREQUENCY
        cx, cy = _center(d)
        px, py = pred
        rx, ry = cx - px, cy - py                 # 予測と観測の差
        tr.x = px + self.alpha * rx
        tr.y = py + self.alpha * ry
        if dt > 0:
            tr.vx += self.beta * rx / dt
            tr.vy += self.beta * ry / dt
        x1, y1, x2, y2 = d.xyxy
        # 大きさは遅れても困らないので強めにならす (枠の辺の揺れを抑える)
        tr.w += self.size_alpha * ((x2 - x1) - tr.w)
        tr.h += self.size_alpha * ((y2 - y1) - tr.h)
        tr.t_qpc = t
        tr.conf = getattr(d, "conf", tr.conf)
        tr.raw_x, tr.raw_y = cx, cy
        tr.hist.append((t, cx, cy, x2 - x1, y2 - y1))
        tr.hits += 1
        tr.misses = 0

    def predict_all(self, t_qpc: Optional[int] = None, extra_ms: float = 0.0
                    ) -> List[Tuple[Track, Tuple[float, float]]]:
        """確定した追跡ごとに (Track, 推定中心) を返す。

        t_qpc:    予測先の時刻 (QPC)。None なら今。
        extra_ms: t_qpc にさらに足す遅延 (ここでは測れない下流の遅延)。
        """
        if t_qpc is None:
            t_qpc = qpc_now()
        t_qpc += int(extra_ms * 1e-3 * QPC_FREQUENCY)
        return [(tr, tr.predict(t_qpc, self.max_lead_ms))
                for tr in self.tracks
                if tr.hits >= self.min_hits and tr.misses == 0]

    def reset(self):
        self.tracks.clear()


def _center(d) -> Tuple[float, float]:
    x1, y1, x2, y2 = d.xyxy
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


# ─── 遅延補正モード (スレッドセーフ) ─────────────────────────────────────────
@dataclass(frozen=True)
class CompensatedBox:
    """遅延補正した枠。Detection と同じ xyxy / conf / cls / name を持つ。"""
    xyxy: Tuple[int, int, int, int]     # 予測先時刻の推定位置
    conf: float
    cls: int
    name: str
    track_id: int
    lead: Tuple[float, float]           # 補正量 (px) = 推定中心 − 最後の観測中心
    lead_ms: float                      # 何 ms 先まで進めたか (頭打ち後)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class LatencyCompensator:
    """YOLO の検出結果を「今」の位置に補正する。

        comp = ketsugecam.LatencyCompensator()
        # 検出スレッド
        comp.update(model.predict(frame, info), info)
        # 使う側 (別スレッドでよい)
        for b in comp.boxes():           # 呼んだ瞬間 + extra_ms の推定位置
            b.xyxy, b.lead

    補正する遅延 = そのフレームの present から boxes() を呼んだ瞬間まで。
    キャプチャ・待ち・推論・受け渡しはすべて毎回の実測 (QPC) で入る。
    extra_ms はその後ろの、ここでは測れない遅延 (表示・入力の反映) を足す分。

    update() と boxes() は別スレッドから呼んでよい。
    """

    def __init__(self, extra_ms: float = 0.0, enabled: bool = True,
                 min_speed: float = 30.0, auto_bias: bool = True,
                 bias_step_ms: float = 0.02, soft_deadband: bool = True,
                 still_ms: float = 100.0, still_px: float = 1.5,
                 hold_px: float = 0.35,
                 **tracker_kwargs):
        """
        still_ms / still_px: 静止判定。直近 still_ms の検出中心に直線を当てはめ、
                    その間の移動量が x/y とも still_px 以下なら止まっているとみなす
                    (抜けるのは 2*still_px を超えたとき。境目でのばたつき防止)。
                    止まっている間は窓内の検出の平均 (中心・大きさ) を出し、
                    先読みもしない → 検出の揺れが平均で消えて枠が動かない。
                    still_ms=0 で無効。
        hold_px:    出力の整数化に粘りを持たせる。表示中の値から 0.5+hold_px 以上
                    離れたときだけ動かす (x.5 付近で 1px 行き来するのを防ぐ)。0 で無効。
        min_speed:  これより遅い (px/s) 対象は止まっているとみなし、進めない。
                    検出の揺れから出る見かけの速度で、止まった対象が揺れるのを防ぐ。
        soft_deadband: min_speed 〜 2*min_speed で補正量を徐々に効かせる。
                    しきい値をまたいだ瞬間に枠が跳ばない。
        tracker_kwargs: Tracker の引数。既定は枠の揺れを抑える値
                    (alpha=0.3, beta=0.05, size_alpha=0.2)。
                    合成データで、揺れ (表示フレーム間の辺の移動) が約半分、
                    位置のずれは同等以下になる組み合わせ。
        auto_bias:  答え合わせで測った予測器の時間ずれ (進みすぎ/遅れ) を
                    少しずつ打ち消す。1件ごとにずれの符号だけ見て bias_step_ms
                    動かすので、中央値に収束し、止まった直後などの大きな外れに
                    引っ張られない。
        """
        self.extra_ms = extra_ms
        self.enabled = enabled
        self.min_speed = min_speed
        self.soft_deadband = soft_deadband
        self.auto_bias = auto_bias
        self.bias_step_s = bias_step_ms * 1e-3
        self._bias_s = 0.0                   # 予測先から差し引く時間 (自動補正量)
        self.still_ms = still_ms
        self.still_px = still_px
        self.hold_px = hold_px
        self._still_ids = set()              # 静止中とみなしている track_id
        self._shown = {}                     # track_id -> 表示中の [x1, y1, x2, y2] (int)
        tracker_kwargs.setdefault("alpha", 0.3)
        tracker_kwargs.setdefault("beta", 0.05)
        tracker_kwargs.setdefault("size_alpha", 0.2)
        self._tracker = Tracker(**tracker_kwargs)
        self._lock = threading.Lock()
        # 合わせ精度の検証用
        self._pending = deque(maxlen=4096)   # boxes() で出した予測 (後で答え合わせ)
        self._hist = {}                      # track_id -> deque[(present, raw_x, raw_y)]
        self._errs = deque(maxlen=2000)      # 答え合わせ済みの誤差

    def update(self, detections: Sequence, info: FrameInfo) -> None:
        with self._lock:
            tracks = self._tracker.update(detections, info)
            if not info.is_new:
                return
            t = Tracker.frame_time(info)
            alive = set()
            for tr in tracks:
                alive.add(tr.id)
                if tr.misses == 0 and tr.t_qpc == t:
                    self._hist.setdefault(tr.id, deque(maxlen=64)).append(
                        (t, tr.raw_x, tr.raw_y))
            for tid in [k for k in self._hist if k not in alive]:
                del self._hist[tid]
            for tid in [k for k in self._shown if k not in alive]:
                del self._shown[tid]
            self._still_ids &= alive
            self._score()

    def _still(self, tr: Track) -> Optional[Tuple[float, float, float, float]]:
        """止まっているなら窓内の平均 (cx, cy, w, h)、動いているなら None。"""
        if self.still_ms <= 0:
            return None
        h = tr.hist
        if not h:
            return None
        t0 = h[-1][0] - int(self.still_ms * 1e-3 * QPC_FREQUENCY)
        pts = [p for p in h if p[0] >= t0]
        n = len(pts)
        if n < 3:
            self._still_ids.discard(tr.id)
            return None
        ts = [(p[0] - pts[0][0]) / QPC_FREQUENCY for p in pts]
        span = ts[-1]
        mt = sum(ts) / n
        stt = sum((t - mt) ** 2 for t in ts)
        mx = sum(p[1] for p in pts) / n
        my = sum(p[2] for p in pts) / n
        if stt > 0:
            # 最小二乗の傾き × 窓の長さ = 窓内で動いた量 (1点の揺れに強い)
            sx = sum((t - mt) * (p[1] - mx) for t, p in zip(ts, pts)) / stt
            sy = sum((t - mt) * (p[2] - my) for t, p in zip(ts, pts)) / stt
            drift = max(abs(sx), abs(sy)) * span
        else:
            drift = 0.0
        lim = self.still_px * (2.0 if tr.id in self._still_ids else 1.0)
        if drift > lim:
            self._still_ids.discard(tr.id)
            return None
        self._still_ids.add(tr.id)
        return (mx, my, sum(p[3] for p in pts) / n, sum(p[4] for p in pts) / n)

    def _hold(self, tid: int, xyxy: Tuple[float, float, float, float]
              ) -> Tuple[int, int, int, int]:
        """整数化に粘りを持たせる (表示中の値から 0.5+hold_px 以上離れたら動かす)。"""
        prev = self._shown.get(tid)
        if prev is None or self.hold_px <= 0:
            out = [round(v) for v in xyxy]
        else:
            th = 0.5 + self.hold_px
            out = [p if abs(v - p) < th else round(v) for v, p in zip(xyxy, prev)]
        self._shown[tid] = out
        return tuple(out)

    def _score(self):
        """出した予測のうち、その時刻をはさむ実フレームが揃ったものを答え合わせする。

        実際の位置 = 予測先時刻の前後のフレーム (present) での検出中心を線形補間。
        """
        gap = 0.05 * QPC_FREQUENCY               # 50ms 以上離れた2フレームでは補間しない
        keep = deque(maxlen=self._pending.maxlen)
        for p in self._pending:
            tid, t, px, py, ox, oy, vx, vy = p
            h = self._hist.get(tid)
            if not h or t < h[0][0]:
                continue                         # 追跡が切れた / 履歴より古い → 捨てる
            if t > h[-1][0]:
                keep.append(p)                   # まだその時刻の画面が来ていない
                continue
            for (ta, xa, ya), (tb, xb, yb) in zip(h, list(h)[1:]):
                if ta <= t <= tb:
                    if tb - ta <= gap:
                        a = (t - ta) / (tb - ta) if tb > ta else 0.0
                        ax, ay = xa + a * (xb - xa), ya + a * (yb - ya)
                        ex, ey = px - ax, py - ay
                        self._errs.append((ex, ey, ox - ax, oy - ay, vx, vy))
                        v2 = vx * vx + vy * vy
                        if self.auto_bias and v2 >= self.min_speed ** 2:
                            # 進行方向のずれの符号で動かす = 中央値を追う (±20ms で頭打ち)
                            along = ex * vx + ey * vy
                            step = self.bias_step_s if along > 0 else -self.bias_step_s
                            self._bias_s = max(-0.02, min(0.02, self._bias_s + step))
                    break
        self._pending = keep

    @property
    def bias_ms(self) -> float:
        """自動補正で予測先から差し引いている時間 (ms)。"""
        return self._bias_s * 1e3

    def alignment(self) -> "Alignment":
        """補正した位置と、その時刻の実際の画面上の位置とのずれ (直近 2000 件)。"""
        with self._lock:
            errs = list(self._errs)
        return Alignment.from_errors(errs)

    def boxes(self, t_qpc: Optional[int] = None,
              extra_ms: Optional[float] = None) -> List[CompensatedBox]:
        """確定した対象の補正済みの枠。enabled=False なら補正量 0 (最後の観測位置)。"""
        if t_qpc is None:
            t_qpc = qpc_now()
        if extra_ms is None:
            extra_ms = self.extra_ms
        target = t_qpc + int(extra_ms * 1e-3 * QPC_FREQUENCY)
        out = []
        with self._lock:
            tr_ = self._tracker
            for tr in tr_.tracks:
                if tr.hits < tr_.min_hits or tr.misses:
                    continue
                w, h = tr.w, tr.h
                still = self._still(tr)
                if still is not None:
                    k = 0.0
                elif self.soft_deadband:
                    # min_speed → 2*min_speed で補正量を 0 → 1 に徐々に効かせる
                    # (しきい値をまたいだ瞬間に枠が跳ばない)
                    k = max(0.0, min(1.0, tr.speed / self.min_speed - 1.0))
                else:
                    k = 1.0 if tr.speed >= self.min_speed else 0.0
                if still is not None:
                    # 止まっている: 窓内の検出の平均で固定 (揺れを平均で消す)
                    px, py, w, h = still
                    lead_ms = 0.0
                elif self.enabled and k > 0.0:
                    aim = target - int(self._bias_s * QPC_FREQUENCY)
                    fx, fy = tr.predict(aim, tr_.max_lead_ms)
                    px, py = tr.x + k * (fx - tr.x), tr.y + k * (fy - tr.y)
                    lead_ms = k * max(0.0, min((aim - tr.t_qpc) / QPC_FREQUENCY * 1e3,
                                               tr_.max_lead_ms))
                elif self.enabled:
                    px, py, lead_ms = tr.x, tr.y, 0.0     # 止まっている
                else:
                    px, py, lead_ms = tr.x, tr.y, 0.0
                self._pending.append((tr.id, target, px, py, tr.raw_x, tr.raw_y,
                                      tr.vx, tr.vy))
                hw, hh = w / 2, h / 2
                out.append(CompensatedBox(
                    xyxy=self._hold(tr.id, (px - hw, py - hh, px + hw, py + hh)),
                    conf=tr.conf, cls=tr.cls, name=tr.name, track_id=tr.id,
                    lead=(px - tr.x, py - tr.y), lead_ms=lead_ms,
                ))
        return out

    def reset(self) -> None:
        with self._lock:
            self._tracker.reset()
            self._pending.clear()
            self._hist.clear()
            self._errs.clear()
            self._still_ids.clear()
            self._shown.clear()
            self._bias_s = 0.0


@dataclass(frozen=True)
class Alignment:
    """補正位置と実際の画面上の位置とのずれ。

    comp_*: 補正した位置の誤差 (px)。raw_*: 補正しなかった場合 (最後の検出位置) の誤差。
    time_bias_ms: 動いている対象での、進行方向のずれを時間に直したもの。
        + なら予測が進みすぎ、- なら遅れ (予測器そのもののクセ)。

    答え合わせは「予測先時刻の画面」とするので、extra_ms の当否はここでは
    分からない (画面より後ろの遅延は画面に写らない)。
    """
    n: int
    comp_p50: float
    comp_p95: float
    raw_p50: float
    raw_p95: float
    time_bias_ms: float
    n_moving: int

    @staticmethod
    def from_errors(errs, min_speed: float = 30.0) -> "Alignment":
        def pct(xs, q):
            if not xs:
                return float("nan")
            xs = sorted(xs)
            return xs[min(int(len(xs) * q / 100), len(xs) - 1)]
        comp = [(ex * ex + ey * ey) ** 0.5 for ex, ey, *_ in errs]
        raw = [(bx * bx + by * by) ** 0.5 for _, _, bx, by, _, _ in errs]
        # 進行方向のずれ / 速さ = 時間のずれ
        lag = [(ex * vx + ey * vy) / (vx * vx + vy * vy)
               for ex, ey, _, _, vx, vy in errs if (vx * vx + vy * vy) ** 0.5 >= min_speed]
        bias = sorted(lag)[len(lag) // 2] * 1e3 if lag else float("nan")
        return Alignment(len(errs), pct(comp, 50), pct(comp, 95),
                         pct(raw, 50), pct(raw, 95), bias, len(lag))
