"""
ketsugecam.detect
─────────────────
TensorRT YOLO 推論エンジン。ketsugecam のキャプチャ結果 (CUDA tensor) を
そのまま受け取って検出結果を返す。

ふつうは直接触る必要はない (`ketsugecam.run()` が内部で使う)。
自分でループを書きたいときはこう使う:

    import ketsugecam
    from ketsugecam.detect import create_detector

    cam = ketsugecam.create(region=(0, 0, 416, 416))
    det = create_detector("best.engine")
    while True:
        frame = cam.grab()
        if frame is not None:
            for obj in det.detect_objects(frame):
                print(obj.label, obj.probability, obj.rect)

中身は C++ 版 (src/EngineBase.h, EngineFP16.h, EngineFP32.h,
EngineFP16Kernels.cu, YoloDetector.{h,cpp}, yolov8.cpp, YoloV11.cpp,
YoloV26.cpp, YoloPostprocKernels.cu, DetectorFactory.cpp) の移植。

前処理はCUDAカーネルと同じ演算をtorchで再現している:
  - バイリニアのレターボックス (左上詰め・右下ゼロパディング、
    src_xy = dst_xy * invScale、エッジクランプ、half-pixelオフセットなし)
  - BGR(A) -> RGB の並べ替え (出力プレーン 0=R, 1=G, 2=B)
  - チャンネルごとの正規化: out = (v * scale255 - sub[c]) / div[c]
  - HWC -> CHW、エンジンの入力dtype (FP16/FP32) へキャスト

FP16エンジンとFP32エンジンで演算が同じなので、どちらも同じ経路を通る。
デコードも各ヘッドのCUDAカーネルとCPUフォールバックの2実装を
torchのベクトル化1本にまとめてある。
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch

# モデルと設定はこのパッケージ内に同梱されている。
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PKG_DIR, "best.engine")


def resolve_model_path(path: Optional[str]) -> str:
    """モデルパスを解決する: 絶対パスはそのまま、相対パスは
    パッケージ内 -> カレントディレクトリ の順に探す。"""
    if not path:
        return DEFAULT_MODEL_PATH
    if os.path.isabs(path):
        return path
    candidate = os.path.normpath(os.path.join(PKG_DIR, path))
    return candidate if os.path.exists(candidate) else path


def read_model_labels(model_path: str) -> Optional[List[str]]:
    """Ultralyticsが書き込んだメタデータ ([int32 長さ][JSON]) からクラス名を読む。

    これがあるので呼び出し側はモデルのパスだけ渡せばよい。
    メタデータが無ければ None。
    """
    try:
        with open(model_path, "rb") as f:
            head = f.read(4)
            if len(head) < 4:
                return None
            length = struct.unpack("<i", head)[0]
            if not 0 < length < (1 << 20):
                return None
            meta = f.read(length)
            if not meta.startswith(b"{"):
                return None
            names = json.loads(meta.decode("utf-8")).get("names")
    except Exception:
        return None

    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda s: int(s))]
        except (KeyError, ValueError):
            return [str(v) for v in names.values()]
    if isinstance(names, list):
        return [str(v) for v in names]
    return None


# ===========================================================================
# TensorRT engine + fused preprocessing
# (= src/EngineBase.h, EngineFP16.h, EngineFP32.h, EngineFP16Kernels.cu)
# ===========================================================================

class Precision(Enum):
    FP32 = 0
    FP16 = 1
    INT8 = 2


@dataclass
class Options:
    precision: Precision = Precision.FP32
    calibration_data_directory_path: str = ""
    calibration_batch_size: int = 128
    opt_batch_size: int = 1
    max_batch_size: int = 16
    device_index: int = 0


# TensorRT logs every severity through, matching the C++ Logger.
LOGGER = trt.Logger(trt.Logger.VERBOSE)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


def does_file_exist(filepath: str) -> bool:
    return os.path.isfile(filepath)


def fnv1a64_hex_of_file(path: str) -> str:
    """64-bit FNV-1a over the bytes of a file. Used to invalidate the cached
    TensorRT engine plan when the source ONNX changes."""
    try:
        f = open(path, "rb")
    except OSError:
        return "nohash"
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF
    h = 1469598103934665603
    with f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            for byte in chunk:
                h = ((h ^ byte) * prime) & mask
    return f"{h:016x}"


class LetterboxGeometry:
    """Precomputed bilinear sampling grid for one (src, dst) size pair.

    src_xy = dst_xy * invScale with neighbours clamped to the source bounds:
    the exact mapping fusedPreprocBGRtoFP16Kernel uses (top-left aligned, no
    half-pixel offset, zero padding right/bottom).
    """

    def __init__(self, src_w: int, src_h: int, dst_w: int, dst_h: int,
                 device: torch.device) -> None:
        self.src_w, self.src_h = src_w, src_h
        self.dst_w, self.dst_h = dst_w, dst_h

        r = min(dst_w / src_w, dst_h / src_h)
        self.r = r
        self.unpad_w = int(r * src_w)
        self.unpad_h = int(r * src_h)
        self.ratio = 1.0 / r  # bbox_dst * ratio = bbox_src
        inv_scale = 1.0 / r

        fx = torch.arange(self.unpad_w, device=device, dtype=torch.float32) * inv_scale
        fy = torch.arange(self.unpad_h, device=device, dtype=torch.float32) * inv_scale
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        self.ax = (fx - x0).view(1, -1, 1)
        self.ay = (fy - y0).view(-1, 1, 1)
        x0i = x0.to(torch.long)
        y0i = y0.to(torch.long)
        self.x0 = x0i.clamp(0, src_w - 1).view(1, -1)
        self.x1 = (x0i + 1).clamp(0, src_w - 1).view(1, -1)
        self.y0 = y0i.clamp(0, src_h - 1).view(-1, 1)
        self.y1 = (y0i + 1).clamp(0, src_h - 1).view(-1, 1)


def fused_preproc(src: torch.Tensor, dst: torch.Tensor, geom: LetterboxGeometry,
                  scale255: float, sub: torch.Tensor, div: torch.Tensor) -> None:
    """Fused uint8 BGR(A) -> normalized CHW, as one torch expression.

    Reproduces EngineFP16Kernels.cu: bilinear letterbox resize, BGR to RGB
    swap, per-channel (v * scale255 - sub) / div, HWC to CHW, cast to the
    engine input dtype.

    src: (H, W, 3|4) uint8 BGR(A) CUDA tensor. dst: (3, dstH, dstW), prezeroed.
    """
    uh, uw = geom.unpad_h, geom.unpad_w

    # Four edge-clamped neighbour gathers. Only the first three bytes of each
    # pixel are read, so BGRA alpha is ignored exactly like the CUDA kernel.
    p00 = src[geom.y0, geom.x0, :3].float()
    p01 = src[geom.y0, geom.x1, :3].float()
    p10 = src[geom.y1, geom.x0, :3].float()
    p11 = src[geom.y1, geom.x1, :3].float()

    top = p00 + (p01 - p00) * geom.ax
    bot = p10 + (p11 - p10) * geom.ax
    val = top + (bot - top) * geom.ay

    val = val.flip(-1)                      # BGR -> RGB
    val = (val * scale255 - sub) / div

    dst[:, :uh, :uw] = val.permute(2, 0, 1).to(dst.dtype)


class TrtEngine:
    """Engine build + cache + load and a persistent device I/O binding set.

    Covers what the C++ splits across EngineFP32 and EngineFP16: the fused
    preprocessing above feeds both FP16 and FP32 engines, since the arithmetic
    is identical (the C++ FP32 path went through cv::cuda resize + split, which
    the pip OpenCV build does not ship and which computes the same geometry).
    """

    def __init__(self, options: Options) -> None:
        self.m_options = options
        self._onnx_hash = ""
        self._runtime: Optional[trt.Runtime] = None
        self._engine = None
        self._context = None

        self._io_tensor_names: List[str] = []
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._buffers: Dict[str, torch.Tensor] = {}

        self._input_dims: List[Tuple[int, int, int]] = []   # (C, H, W)
        self._output_dims: List[Tuple[int, ...]] = []
        self._output_lengths: List[int] = []

        self._engine_is_fp32 = False
        self._input_is_float = False

        self.device = torch.device("cuda", options.device_index)
        torch.cuda.set_device(self.device)
        # Per-engine CUDA stream reused across inferences (= EngineBase m_stream).
        self.stream = torch.cuda.Stream(device=self.device)

        self._geom: Optional[LetterboxGeometry] = None
        self._sub = torch.zeros(3, device=self.device)
        self._div = torch.ones(3, device=self.device)
        self._scale255 = 1.0 / 255.0

    # -- Properties ---------------------------------------------------------

    @property
    def engine_is_fp32(self) -> bool:
        return self._engine_is_fp32

    @property
    def input_dims(self) -> List[Tuple[int, int, int]]:
        return self._input_dims

    @property
    def output_dims(self) -> List[Tuple[int, ...]]:
        return self._output_dims

    def output_tensor(self, output_idx: int) -> torch.Tensor:
        return self._buffers[self._output_names[output_idx]]

    # -- Build / load -------------------------------------------------------

    def build_load_network(self, onnx_model_path: str,
                           sub_vals: Sequence[float] = (0.0, 0.0, 0.0),
                           div_vals: Sequence[float] = (1.0, 1.0, 1.0),
                           normalize: bool = True) -> bool:
        # A prebuilt serialized engine is loaded as-is (no ONNX, no cache-name lookup).
        if len(onnx_model_path) > 7 and onnx_model_path.endswith(".engine"):
            return self.load_network(onnx_model_path, sub_vals, div_vals, normalize)

        self._onnx_hash = fnv1a64_hex_of_file(onnx_model_path)
        engine_name = self._serialize_engine_options(onnx_model_path)
        print(f"Searching for engine file with name: {engine_name}")

        if does_file_exist(engine_name):
            print("Engine found, not regenerating...")
        else:
            if not does_file_exist(onnx_model_path):
                raise RuntimeError(f"Could not find onnx model at path: {onnx_model_path}")
            print("Engine not found, generating. This could take a while...")
            if not self._build_serialized(onnx_model_path, engine_name):
                return False

        return self.load_network(engine_name, sub_vals, div_vals, normalize)

    def load_network(self, trt_model_path: str,
                     sub_vals: Sequence[float] = (0.0, 0.0, 0.0),
                     div_vals: Sequence[float] = (1.0, 1.0, 1.0),
                     normalize: bool = True) -> bool:
        if not does_file_exist(trt_model_path):
            print(f"Error, unable to read TensorRT model at path: {trt_model_path}")
            return False

        print(f"Loading TensorRT engine file at path: {trt_model_path}")
        with open(trt_model_path, "rb") as f:
            buffer = f.read()

        # Ultralytics-exported engines prepend [int32 length][JSON metadata]
        # before the TRT plan.
        plan_offset = 0
        if len(buffer) > 8 and buffer[4:5] == b"{":
            meta_len = struct.unpack("<i", buffer[:4])[0]
            if meta_len > 0 and meta_len + 4 < len(buffer):
                plan_offset = meta_len + 4

        self._runtime = trt.Runtime(LOGGER)
        self._engine = self._runtime.deserialize_cuda_engine(buffer[plan_offset:])
        if self._engine is None:
            return False

        self._context = self._engine.create_execution_context()
        if self._context is None:
            return False
        print("[TrtEngine] Execution context created, setting up I/O...")

        self._buffers.clear()
        self._io_tensor_names.clear()
        self._input_names.clear()
        self._output_names.clear()
        self._input_dims.clear()
        self._output_dims.clear()
        self._output_lengths.clear()

        n_io = self._engine.num_io_tensors
        print(f"[TrtEngine] {n_io} I/O tensors")

        # Detect actual tensor precision: the ONNX may be exported as FP16 but
        # TensorRT may upcast to FP32 internally.
        engine_is_fp16 = all(
            self._engine.get_tensor_dtype(self._engine.get_tensor_name(i)) == trt.DataType.HALF
            for i in range(n_io)
        )
        self._engine_is_fp32 = not engine_is_fp16
        if self._engine_is_fp32 and self.m_options.precision == Precision.FP16:
            print("[TrtEngine] TensorRT upcast FP16 ONNX to FP32 engine, using FP32 buffers")

        for i in range(n_io):
            name = self._engine.get_tensor_name(i)
            self._io_tensor_names.append(name)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = self._engine.get_tensor_dtype(name)
            torch_dtype = _TRT_TO_TORCH[dtype]

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                c = shape[1]
                h = shape[2] if shape[2] > 0 else 640
                w = shape[3] if shape[3] > 0 else 640
                self._input_dims.append((c, h, w))
                self._input_is_float = dtype == trt.DataType.FLOAT
                self._buffers[name] = torch.zeros(
                    (self.m_options.max_batch_size, c, h, w),
                    dtype=torch_dtype, device=self.device)
            else:
                self._output_names.append(name)
                self._output_dims.append(shape)
                length = 1
                for d in shape[1:]:
                    length *= d
                self._output_lengths.append(length)
                self._buffers[name] = torch.zeros(
                    (self.m_options.max_batch_size * length,),
                    dtype=torch_dtype, device=self.device)

        for name in self._io_tensor_names:
            self._context.set_tensor_address(name, int(self._buffers[name].data_ptr()))

        self._sub = torch.tensor([float(v) for v in sub_vals], device=self.device)
        self._div = torch.tensor([v if v != 0.0 else 1.0 for v in map(float, div_vals)],
                                 device=self.device)
        self._scale255 = (1.0 / 255.0) if normalize else 1.0

        torch.cuda.current_stream(self.device).synchronize()
        return True

    # -- Inference (= runInferenceFromBGR) ----------------------------------

    def run_inference_from_bgr(self, bgr: torch.Tensor) -> float:
        """Fused preproc into the engine input buffer + enqueue, both on the
        caller's current stream. Returns immediately after enqueue (no sync, no
        D2H) so the caller can chain its own postprocess on the same stream.

        Returns the postprocess scale factor: bbox_dst * ratio = bbox_src.
        """
        if bgr.numel() == 0 or bgr.dtype != torch.uint8 or bgr.shape[2] not in (3, 4):
            raise RuntimeError("run_inference_from_bgr expects a non-empty "
                               "(H, W, 3|4) uint8 BGR(A) CUDA tensor")
        if len(self._input_dims) != 1:
            raise RuntimeError("run_inference_from_bgr requires a single input binding")

        dst_c, dst_h, dst_w = self._input_dims[0]
        if dst_c != 3:
            raise RuntimeError("run_inference_from_bgr only supports 3-channel input")

        src_h, src_w = int(bgr.shape[0]), int(bgr.shape[1])
        geom = self._geom
        if (geom is None or geom.src_w != src_w or geom.src_h != src_h
                or geom.dst_w != dst_w or geom.dst_h != dst_h):
            geom = LetterboxGeometry(src_w, src_h, dst_w, dst_h, self.device)
            self._geom = geom
            self._buffers[self._input_names[0]].zero_()  # padding region stays zero

        fused_preproc(bgr, self._buffers[self._input_names[0]][0], geom,
                      self._scale255, self._sub, self._div)

        self._context.set_input_shape(self._input_names[0], (1, dst_c, dst_h, dst_w))

        handle = torch.cuda.current_stream(self.device).cuda_stream
        if not self._context.execute_async_v3(handle):
            raise RuntimeError("TensorRT enqueueV3 failed")

        return geom.ratio

    # -- Engine cache name / build ------------------------------------------

    def _precision_suffix(self) -> str:
        return "fp16" if self.m_options.precision == Precision.FP16 else "fp32"

    def _serialize_engine_options(self, onnx_model_path: str) -> str:
        # Keep the model's directory in the name so the cached plan lands next
        # to the ONNX (the C++ wrote it into the working directory instead).
        engine_name = os.path.splitext(onnx_model_path)[0] + ".engine"

        device_name = "".join(torch.cuda.get_device_name(
            self.m_options.device_index).split())

        engine_name += "." + device_name
        engine_name += "." + self._precision_suffix()
        engine_name += "." + str(self.m_options.max_batch_size)
        engine_name += "." + str(self.m_options.opt_batch_size)
        if self._onnx_hash:
            engine_name += "." + self._onnx_hash
        return engine_name

    def _build_serialized(self, onnx_model_path: str, engine_name: str) -> bool:
        builder = trt.Builder(LOGGER)
        network = builder.create_network()
        parser = trt.OnnxParser(network, LOGGER)

        with open(onnx_model_path, "rb") as f:
            model_bytes = f.read()
        if not parser.parse(model_bytes):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

        num_inputs = network.num_inputs
        if num_inputs < 1:
            raise RuntimeError("Error, model needs at least 1 input!")

        config = builder.create_builder_config()
        if self.m_options.precision == Precision.FP16:
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("GPU does not support FP16")
            config.set_flag(trt.BuilderFlag.FP16)
            if builder.platform_has_tf32:
                config.set_flag(trt.BuilderFlag.TF32)
        elif self.m_options.precision == Precision.INT8:
            raise RuntimeError("INT8 engine not implemented yet")

        profile = builder.create_optimization_profile()
        for i in range(num_inputs):
            inp = network.get_input(i)
            dims = inp.shape
            input_c = dims[1]
            # ONNX models may export with dynamic spatial dims (-1); the
            # optimization profile needs concrete values, so default to 640.
            input_h = dims[2] if dims[2] > 0 else 640
            input_w = dims[3] if dims[3] > 0 else 640
            profile.set_shape(
                inp.name,
                (1, input_c, input_h, input_w),
                (self.m_options.opt_batch_size, input_c, input_h, input_w),
                (self.m_options.max_batch_size, input_c, input_h, input_w),
            )
        config.add_optimization_profile(profile)

        profile_stream = torch.cuda.Stream(device=self.device)
        config.profile_stream = profile_stream.cuda_stream

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            return False

        with open(engine_name, "wb") as f:
            f.write(bytes(plan))
        return True


# ===========================================================================
# Detector (= src/YoloDetector.{h,cpp}, yolov8.cpp, YoloV11.cpp, YoloV26.cpp,
#              YoloPostprocKernels.cu, DetectorFactory.cpp)
# ===========================================================================

@dataclass
class Object:
    label: int = 0
    probability: float = 0.0
    rect: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h
    box_mask: Optional[np.ndarray] = None
    kps: List[float] = field(default_factory=list)


class YoloVersion(Enum):
    AUTO = 0  # auto-detect from model file
    V8 = 1    # output (1, 4+C, 8400) - no DFL, NMS required
    V11 = 2   # output (1, 4*regMax+C, 8400) - DFL decode, NMS required
    V26 = 3   # output (1, 300, 6) - end-to-end, no NMS


@dataclass
class YoloConfig:
    precision: Precision = Precision.FP16
    calibration_data_directory: str = ""
    probability_threshold: float = 0.25
    nms_threshold: float = 0.65
    top_k: int = 100
    # Segmentation
    seg_channels: int = 32
    seg_h: int = 160
    seg_w: int = 160
    segmentation_threshold: float = 0.5
    # Pose
    num_kps: int = 17
    kps_threshold: float = 0.5
    # Class names. Left empty, they come from the model metadata
    # (read_model_labels), and failing that from the output channel count.
    class_names: List[str] = field(default_factory=list)
    # NMS across classes instead of per class (ultralytics agnostic_nms)
    agnostic_nms: bool = False
    # V11 DFL
    reg_max: int = 16
    # V26
    max_detections_v26: int = 300


class YoloDetector:
    """Engine management, CUDA graph, GPU buffers, drawing, auto-detection.

    Subclasses implement the version-specific decode. The C++ ships two decode
    implementations per version (a CUDA kernel for FP16 engines, a CPU loop for
    FP32 ones) that compute the same thing; one vectorized torch decode covers
    both, so FP16 and FP32 engines take the same path.

    Survivor records keep the C++ layout: (x0, y0, x1, y1, score, label).
    """

    K_MAX_SURVIVORS = 1024
    K_SURVIVOR_STRIDE = 6

    # Normalization constants (used by the fused preproc)
    SUB_VALS = (0.0, 0.0, 0.0)
    DIV_VALS = (1.0, 1.0, 1.0)
    NORMALIZE = True

    # Colours (BGR, 0..1 - scaled where the C++ scales them)
    BOX_COLORS = [
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 1.0),
    ]

    KPS_COLORS = [
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
        (255, 128, 0), (255, 128, 0), (51, 153, 255), (51, 153, 255),
        (51, 153, 255), (51, 153, 255), (51, 153, 255), (128, 0, 128),
        (255, 0, 255),
    ]

    SKELETON = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12),
        (7, 13), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
        (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
    ]

    LIMB_COLORS = [
        (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
        (255, 51, 255), (255, 51, 255), (255, 51, 255), (255, 128, 0),
        (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 255, 0), (0, 255, 0), (0, 255, 0),
    ]

    def __init__(self, onnx_model_path: str, config: YoloConfig,
                 version: YoloVersion) -> None:
        self.PROBABILITY_THRESHOLD = config.probability_threshold
        self.NMS_THRESHOLD = config.nms_threshold
        self.TOP_K = config.top_k
        self.SEG_CHANNELS = config.seg_channels
        self.SEG_H = config.seg_h
        self.SEG_W = config.seg_w
        self.SEGMENTATION_THRESHOLD = config.segmentation_threshold
        self.CLASS_NAMES = list(config.class_names)
        self.NUM_KPS = config.num_kps
        self.KPS_THRESHOLD = config.kps_threshold
        self.AGNOSTIC_NMS = config.agnostic_nms
        self.REG_MAX = config.reg_max
        self.MAX_DETECTIONS_V26 = config.max_detections_v26
        self.m_version = version

        options = Options(
            precision=config.precision,
            calibration_data_directory_path=config.calibration_data_directory,
            opt_batch_size=1,
            max_batch_size=1,
        )
        if options.precision == Precision.INT8 and not options.calibration_data_directory_path:
            raise RuntimeError("Error: Must supply calibration data path for INT8 calibration")

        self._engine = TrtEngine(options)
        print("Building or loading TensorRT engine...")
        if not self._engine.build_load_network(onnx_model_path, self.SUB_VALS,
                                               self.DIV_VALS, self.NORMALIZE):
            raise RuntimeError("Error: Unable to build or load the TensorRT engine. "
                               "Try increasing TensorRT log severity to kVERBOSE.")
        print("TensorRT engine built and loaded!")

        self._output_is_fp32 = self._engine.engine_is_fp32
        if self._output_is_fp32:
            print("[YoloDetector] Engine has FP32 output")

        # Cached output dimensions - never change after engine construction.
        self.m_num_anchors = 0
        self.m_num_classes = 0
        self.m_num_channels = 0
        if self._engine.output_dims:
            od = self._engine.output_dims[0]
            self.m_num_anchors = int(od[2])
            self.m_num_channels = int(od[1])

            if self.m_version == YoloVersion.V26:
                self.m_num_classes = len(self.CLASS_NAMES)
                print(f"[YoloDetector] V26 output dims: ({od[0]}, {od[1]}, {od[2]}) "
                      f" maxDetectionsV26={self.MAX_DETECTIONS_V26}")
            elif self.m_version == YoloVersion.V11:
                self.m_num_classes = self.m_num_channels - 4 * self.REG_MAX
            else:
                self.m_num_classes = self.m_num_channels - 4

        if not self.CLASS_NAMES and self.m_num_classes > 0:
            # No labels anywhere: number them, so drawing still says something.
            self.CLASS_NAMES = [str(i) for i in range(self.m_num_classes)]

        # Postprocess geometry (set by the first inference, constant afterwards).
        self.m_ratio = 1.0
        self.m_img_width = 0.0
        self.m_img_height = 0.0

        # Capture buffer + graph capture state.
        self._src_channels = 0
        self._capture_buf: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_captured = False
        self._graph_disabled = os.environ.get("YOLO_NO_CUDA_GRAPH", "") not in ("", "0")

        self._arange_bins = torch.arange(self.REG_MAX, dtype=torch.float32,
                                         device=self._engine.device).view(1, -1, 1)
        print("[YoloDetector] Constructor complete")

    @property
    def engine(self) -> TrtEngine:
        return self._engine

    # -- Version-specific decode -------------------------------------------

    def _decode_survivors(self) -> torch.Tensor:
        """Threshold + decode the engine output into a compact
        (N, 6) = (x0, y0, x1, y1, score, label) CUDA tensor."""
        raise NotImplementedError

    # -- Inference ----------------------------------------------------------

    def detect_objects(self, input_image_bgr: torch.Tensor) -> List[Object]:
        if not input_image_bgr.is_cuda or input_image_bgr.dtype != torch.uint8:
            raise RuntimeError("Error: detect_objects requires a uint8 CUDA tensor.")

        if self._src_channels == 0:
            channels = int(input_image_bgr.shape[2]) if input_image_bgr.dim() == 3 else 0
            if channels not in (3, 4):
                raise RuntimeError("Error: detect_objects requires (H, W, 3) BGR or "
                                   "(H, W, 4) BGRA uint8 tensors.")
            self._src_channels = channels

        height = int(input_image_bgr.shape[0])
        width = int(input_image_bgr.shape[1])

        with torch.cuda.stream(self._engine.stream):
            buf = self._capture_buf
            if (buf is None or buf.shape[0] != height or buf.shape[1] != width
                    or buf.shape[2] != self._src_channels):
                self._release_graph()
                buf = torch.empty((height, width, self._src_channels),
                                  dtype=torch.uint8, device=self._engine.device)
                self._capture_buf = buf

            # The capture side hands us a view of a reused device buffer, so the
            # frame is copied into our own fixed-address buffer. That is also
            # what makes CUDA graph replay possible.
            buf.copy_(input_image_bgr, non_blocking=True)

            if self._graph_captured:
                self._graph.replay()
            elif self._graph_disabled:
                self._enqueue()
            else:
                self._capture_graph()

            survivors = self._decode_survivors()
            host = survivors.to("cpu")

        return self._postprocess_from_survivors(host.numpy())

    def _enqueue(self) -> None:
        """Fused preproc + enqueueV3 on the engine stream."""
        self.m_ratio = self._engine.run_inference_from_bgr(self._capture_buf)
        self.m_img_width = float(self._capture_buf.shape[1])
        self.m_img_height = float(self._capture_buf.shape[0])

    def _capture_graph(self) -> None:
        """Warm up, then capture preproc + inference into a CUDA graph."""
        self._enqueue()  # Warmup
        self._engine.stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            # thread_local capture mode: the capture thread keeps launching its
            # own copies on another stream while we capture (= the C++
            # cudaStreamCaptureModeThreadLocal).
            with torch.cuda.graph(graph, stream=self._engine.stream,
                                  capture_error_mode="thread_local"):
                self._enqueue()
        except Exception as e:  # capture unsupported - fall back to eager launches
            print(f"[YoloDetector] CUDA graph capture failed, running eagerly: {e}")
            self._graph_disabled = True
            self._engine.stream.synchronize()
            self._enqueue()
            return

        self._graph = graph
        self._graph_captured = True
        self._engine.stream.synchronize()
        graph.replay()

    def _release_graph(self) -> None:
        self._graph = None
        self._graph_captured = False

    # -- Shared NMS postprocessing (used by V8 and V11; V26 overrides) ------

    def _postprocess_from_survivors(self, survivors: np.ndarray) -> List[Object]:
        n = min(survivors.shape[0], self.K_MAX_SURVIVORS)
        if n == 0:
            return []

        s = survivors[:n]
        # cv::Rect is integral, and the C++ pushes float rects into a
        # vector<cv::Rect>, so the box coordinates are rounded before NMS.
        x0 = np.rint(s[:, 0]).astype(np.int32)
        y0 = np.rint(s[:, 1]).astype(np.int32)
        x1 = np.rint(s[:, 2]).astype(np.int32)
        y1 = np.rint(s[:, 3]).astype(np.int32)
        bboxes = [(int(a), int(b), int(c - a), int(d - b))
                  for a, b, c, d in zip(x0, y0, x1, y1)]
        scores = [float(v) for v in s[:, 4]]
        labels = [int(v) for v in s[:, 5]]

        if self.AGNOSTIC_NMS:
            # Suppress across classes (ultralytics agnostic_nms).
            indices = cv2.dnn.NMSBoxes(bboxes, scores,
                                       self.PROBABILITY_THRESHOLD,
                                       self.NMS_THRESHOLD)
        else:
            indices = cv2.dnn.NMSBoxesBatched(bboxes, scores, labels,
                                              self.PROBABILITY_THRESHOLD,
                                              self.NMS_THRESHOLD)

        objects: List[Object] = []
        for chosen_idx in np.asarray(indices).reshape(-1):
            if len(objects) >= self.TOP_K:
                break
            i = int(chosen_idx)
            objects.append(Object(label=labels[i], probability=scores[i],
                                  rect=bboxes[i]))
        return objects

    # -- Drawing ------------------------------------------------------------

    def draw_object_labels(self, image: np.ndarray, objects: Sequence[Object],
                           scale: int = 2, square_half_size: int = 0) -> None:
        if objects and objects[0].box_mask is not None:
            mask = image.copy()
            for obj in objects:
                color = self.BOX_COLORS[obj.label % len(self.BOX_COLORS)]
                x, y, w, h = (int(v) for v in obj.rect)
                roi = mask[y:y + h, x:x + w]
                roi[obj.box_mask.astype(bool)] = tuple(c * 255 for c in color)
            cv2.addWeighted(image, 0.5, mask, 0.8, 1, image)

        for obj in objects:
            color = self.BOX_COLORS[obj.label % len(self.BOX_COLORS)]
            rect = tuple(int(v) for v in obj.rect)

            name = (self.CLASS_NAMES[obj.label]
                    if 0 <= obj.label < len(self.CLASS_NAMES) else "")
            text = f"{name} {obj.label}"

            (label_w, label_h), base_line = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale)
            txt_bk_color = tuple(c * 0.7 * 255 for c in color)

            x = int(obj.rect[0])
            y = int(obj.rect[1]) + 1

            cv2.rectangle(image, rect, tuple(c * 255 for c in color), 1)
            cv2.rectangle(image, (x, y, label_w, label_h + base_line),
                          txt_bk_color, -1)
            cv2.putText(image, text, (x, y + label_h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * scale, color, scale)

            if obj.kps:
                kps = obj.kps
                for k in range(self.NUM_KPS + 2):
                    if k < self.NUM_KPS:
                        kps_s = kps[k * 3 + 2]
                        if kps_s > self.KPS_THRESHOLD:
                            cv2.circle(image, (round(kps[k * 3]), round(kps[k * 3 + 1])),
                                       5, self.KPS_COLORS[k], -1)
                    ske = self.SKELETON[k]
                    pos1_s = kps[(ske[0] - 1) * 3 + 2]
                    pos2_s = kps[(ske[1] - 1) * 3 + 2]
                    if pos1_s > self.KPS_THRESHOLD and pos2_s > self.KPS_THRESHOLD:
                        cv2.line(image,
                                 (round(kps[(ske[0] - 1) * 3]), round(kps[(ske[0] - 1) * 3 + 1])),
                                 (round(kps[(ske[1] - 1) * 3]), round(kps[(ske[1] - 1) * 3 + 1])),
                                 self.LIMB_COLORS[k], 2)

    # -- Auto-detection -----------------------------------------------------

    @staticmethod
    def detect_version_from_output(output_dims: Sequence[int],
                                   num_classes: int) -> YoloVersion:
        # outputDims layout: (batch, channels, anchors) or (batch, detections, boxinfo)
        channels = int(output_dims[1])
        anchors = int(output_dims[2])

        # V26 end-to-end: output can be (1, 6, ~300) or (1, ~300, 6).
        # The 6 fields are: (x1, y1, x2, y2, confidence, class_id)
        if channels == 6 and 100 <= anchors <= 500:
            return YoloVersion.V26
        if anchors == 6 and 100 <= channels <= 500:
            return YoloVersion.V26

        # V8 vs V11: output is (1, 4+C, 8400) or (1, 4*regMax+C, 8400).
        bbox_channels = channels - num_classes

        if bbox_channels == 4:
            return YoloVersion.V8

        # If bboxChannels is a multiple of 4 and > 4, it is likely DFL (V11).
        if bbox_channels > 4 and bbox_channels % 4 == 0 and bbox_channels <= 256:
            if bbox_channels // 4 in (8, 16, 32):
                return YoloVersion.V11

        # Can't distinguish - default to V8.
        return YoloVersion.V8

    @staticmethod
    def detect_version_from_filename(model_path: str) -> YoloVersion:
        name = model_path.replace("\\", "/")
        name = name[name.rfind("/") + 1:].lower()

        # Order matters: check more specific patterns first.
        if "yolo26" in name or "yolov26" in name:
            return YoloVersion.V26
        if "yolo11" in name or "yolov11" in name:
            return YoloVersion.V11
        if "yolo8" in name or "yolov8" in name:
            return YoloVersion.V8
        return YoloVersion.AUTO  # no signal from filename


class YoloV8(YoloDetector):
    """YOLOv8 - standard anchor decode (no DFL), NMS required."""

    def __init__(self, onnx_model_path: str, config: YoloConfig) -> None:
        super().__init__(onnx_model_path, config, YoloVersion.V8)

    def _decode_survivors(self) -> torch.Tensor:
        n_ch, n_a = self.m_num_channels, self.m_num_anchors
        out = self._engine.output_tensor(0)[:n_ch * n_a].view(n_ch, n_a).float()

        # Find best class score.
        best_score, best_label = out[4:4 + self.m_num_classes].max(dim=0)

        # Decode xywh -> xyxy in source-image space.
        x, y, w, h = out[0], out[1], out[2], out[3]
        x0 = ((x - 0.5 * w) * self.m_ratio).clamp(0.0, self.m_img_width)
        y0 = ((y - 0.5 * h) * self.m_ratio).clamp(0.0, self.m_img_height)
        x1 = ((x + 0.5 * w) * self.m_ratio).clamp(0.0, self.m_img_width)
        y1 = ((y + 0.5 * h) * self.m_ratio).clamp(0.0, self.m_img_height)

        keep = best_score > self.PROBABILITY_THRESHOLD
        survivors = torch.stack((x0, y0, x1, y1, best_score,
                                 best_label.float()), dim=1)[keep]
        return survivors[:self.K_MAX_SURVIVORS]


class YoloV11(YoloDetector):
    """YOLOv11 - DFL-aware anchor decode, NMS required."""

    def __init__(self, onnx_model_path: str, config: YoloConfig) -> None:
        super().__init__(onnx_model_path, config, YoloVersion.V11)

    def _decode_survivors(self) -> torch.Tensor:
        n_ch, n_a = self.m_num_channels, self.m_num_anchors
        reg_max = self.REG_MAX
        out = self._engine.output_tensor(0)[:n_ch * n_a].view(n_ch, n_a).float()

        # DFL decode: each of the 4 coordinates is distributed over regMax bins.
        # coord = sum(bin * softmax(bin))
        dist = out[:4 * reg_max].view(4, reg_max, n_a)
        coords = (torch.softmax(dist, dim=1) * self._arange_bins).sum(dim=1)
        x, y, w, h = coords[0], coords[1], coords[2], coords[3]

        class_offset = 4 * reg_max
        best_score, best_label = out[class_offset:class_offset + self.m_num_classes].max(dim=0)

        x0 = ((x - 0.5 * w) * self.m_ratio).clamp(0.0, self.m_img_width)
        y0 = ((y - 0.5 * h) * self.m_ratio).clamp(0.0, self.m_img_height)
        x1 = ((x + 0.5 * w) * self.m_ratio).clamp(0.0, self.m_img_width)
        y1 = ((y + 0.5 * h) * self.m_ratio).clamp(0.0, self.m_img_height)

        keep = best_score > self.PROBABILITY_THRESHOLD
        survivors = torch.stack((x0, y0, x1, y1, best_score,
                                 best_label.float()), dim=1)[keep]
        return survivors[:self.K_MAX_SURVIVORS]


class YoloV26(YoloDetector):
    """YOLOv26 - end-to-end, NMS-free, threshold only."""

    def __init__(self, onnx_model_path: str, config: YoloConfig) -> None:
        super().__init__(onnx_model_path, config, YoloVersion.V26)

    def _decode_survivors(self) -> torch.Tensor:
        # Output layout: (1, maxDetections, 6) - 6 fields contiguous per
        # detection, already decoded as (x1, y1, x2, y2, confidence, class_id).
        buf = self._engine.output_tensor(0)
        max_det = min(self.MAX_DETECTIONS_V26, buf.numel() // 6)
        out = buf[:max_det * 6].view(max_det, 6).float()

        keep = out[:, 4] > self.PROBABILITY_THRESHOLD
        return out[keep][:self.K_MAX_SURVIVORS]

    def _postprocess_from_survivors(self, survivors: np.ndarray) -> List[Object]:
        # The decode already applied the confidence threshold and wrote
        # survivors in (x1, y1, x2, y2, confidence, class_id) format.
        # No NMS, no re-check.
        n = min(survivors.shape[0], self.K_MAX_SURVIVORS)

        objects: List[Object] = []
        for i in range(n):
            if len(objects) >= self.TOP_K:
                break
            s = survivors[i]
            objects.append(Object(
                label=int(s[5]),
                probability=float(s[4]),
                rect=(float(s[0]), float(s[1]),
                      float(s[2]) - float(s[0]), float(s[3]) - float(s[1])),
            ))
        return objects


_BY_VERSION = {
    YoloVersion.V8: YoloV8,
    YoloVersion.V11: YoloV11,
    YoloVersion.V26: YoloV26,
}


def create_detector(model_path: Optional[str] = None,
                    config: Optional[YoloConfig] = None,
                    version: YoloVersion = YoloVersion.AUTO) -> YoloDetector:
    """Create a detector for the given model (= DetectorFactory::create).

    Only the model path is required: relative paths resolve inside the package
    first, the class names come from the model's own metadata, and everything
    else uses the YoloConfig defaults.

    If `version` is not AUTO it is used directly. When AUTO, the version is
    inferred from the filename first, then from the output shape after load.
    The engine is built and loaded here, so the detector comes back ready.
    """
    onnx_model_path = resolve_model_path(model_path)
    config = config or YoloConfig()
    if not config.class_names:
        labels = read_model_labels(onnx_model_path)
        if labels:
            print(f"[detect] クラス名をモデルから読み込み: {labels}")
            config.class_names = labels
    if version != YoloVersion.AUTO:
        return _BY_VERSION[version](onnx_model_path, config)

    # First pass: filename-based detection. Fast, and works before the engine
    # is built.
    file_version = YoloDetector.detect_version_from_filename(onnx_model_path)
    if file_version != YoloVersion.AUTO:
        name = {YoloVersion.V11: "YOLOv11",
                YoloVersion.V26: "YOLOv26"}.get(file_version, "YOLOv8")
        print(f"[DetectorFactory] Detected {name} from model filename.")
        return _BY_VERSION[file_version](onnx_model_path, config)

    # Second pass: load the engine and inspect the output shape. Handles
    # renamed files and non-standard naming.
    temp = YoloV8(onnx_model_path, config)
    output_dims = temp.engine.output_dims
    shape_version = (YoloDetector.detect_version_from_output(
        output_dims[0], len(config.class_names)) if output_dims else YoloVersion.V8)

    if shape_version == YoloVersion.V8:
        return temp  # correct already

    # Shape says V11 or V26 - discard temp and re-create. The engine plan is
    # cached, so re-creation just deserializes.
    print(f"[DetectorFactory] Auto-detected "
          f"{'YOLOv11' if shape_version == YoloVersion.V11 else 'YOLOv26'} "
          f"from output shape.")
    return _BY_VERSION[shape_version](onnx_model_path, config)




# ===========================================================================
# 呼び出し側向けの薄いラッパ (ultralytics の YOLO().predict() に寄せた形)
# ===========================================================================

@dataclass
class Detection:
    """検出1件。`xyxy` はキャプチャしたROI内のピクセル座標 (int)。"""

    xyxy: Tuple[int, int, int, int]
    conf: float
    cls: int
    name: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)


class Detector:
    """ketsugecam のフレームをそのまま食わせる TensorRT YOLO 検出器。

        import ketsugecam

        cam   = ketsugecam.create(region=(752, 332, 1168, 748))
        model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)

        frame = cam.grab()                 # torch.Tensor [H, W, 3] uint8 BGR (CUDA)
        for d in model.predict(frame):     # 前処理・デコード・NMSは内蔵
            print(d.xyxy, d.conf, d.name, d.center)

    `conf` / `iou` / `max_det` は属性としてあとから書き換えてもよい。
    """

    def __init__(self, model_path: Optional[str] = None,
                 conf: float = 0.25,
                 iou: float = 0.65,
                 max_det: int = 100,
                 agnostic_nms: bool = False,
                 version: YoloVersion = YoloVersion.AUTO) -> None:
        config = YoloConfig(probability_threshold=float(conf),
                            nms_threshold=float(iou),
                            top_k=int(max_det),
                            agnostic_nms=bool(agnostic_nms))
        self.yolo = create_detector(model_path, config, version)

    # -- ultralytics 風のプロパティ -----------------------------------------

    @property
    def names(self) -> List[str]:
        """クラス名 (モデルのメタデータから読んだもの)。"""
        return self.yolo.CLASS_NAMES

    @property
    def imgsz(self) -> Tuple[int, int]:
        """モデルの入力サイズ (幅, 高さ)。"""
        _, h, w = self.yolo.engine.input_dims[0]
        return (w, h)

    @property
    def version(self) -> YoloVersion:
        return self.yolo.m_version

    @property
    def conf(self) -> float:
        return self.yolo.PROBABILITY_THRESHOLD

    @conf.setter
    def conf(self, value: float) -> None:
        self.yolo.PROBABILITY_THRESHOLD = float(value)

    @property
    def iou(self) -> float:
        return self.yolo.NMS_THRESHOLD

    @iou.setter
    def iou(self, value: float) -> None:
        self.yolo.NMS_THRESHOLD = float(value)

    @property
    def max_det(self) -> int:
        return self.yolo.TOP_K

    @max_det.setter
    def max_det(self, value: int) -> None:
        self.yolo.TOP_K = int(value)

    # -- 推論 ---------------------------------------------------------------

    def predict(self, frame) -> List[Detection]:
        """1フレーム推論する。

        frame: ketsugecam が返す torch.Tensor [H, W, 3|4] uint8 BGR(A) (CUDA)、
               または numpy の BGR 画像 (自動でGPUへ上げる)。
               permute / float / 正規化 / RGB変換は内部でやるので不要。
        """
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(np.ascontiguousarray(frame)).to(
                self.yolo.engine.device)
        elif not frame.is_cuda:
            frame = frame.to(self.yolo.engine.device)

        names = self.yolo.CLASS_NAMES
        out: List[Detection] = []
        for obj in self.yolo.detect_objects(frame):
            x, y, w, h = obj.rect
            out.append(Detection(
                xyxy=(int(x), int(y), int(x + w), int(y + h)),
                conf=float(obj.probability),
                cls=int(obj.label),
                name=names[obj.label] if 0 <= obj.label < len(names) else "",
            ))
        return out

    __call__ = predict

    # -- 描画 ---------------------------------------------------------------

    def draw(self, image: np.ndarray, detections: Sequence[Detection],
             color: Tuple[int, int, int] = (0, 255, 0),
             show_conf: bool = True, show_name: bool = False,
             center_dot: bool = True) -> np.ndarray:
        """numpy BGR 画像に枠を描き込む (その場で書き換えて返す)。"""
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = ""
            if show_name and d.name:
                label = d.name
            if show_conf:
                label = f"{label} {d.conf:.2f}".strip()
            if label:
                cv2.putText(image, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if center_dot:
                cv2.circle(image, d.center, 3, (0, 0, 255), -1)
        return image


def _resolve_from_caller(path: str, caller_file: Optional[str]) -> str:
    """相対パスを 呼び出し元スクリプトのフォルダ -> カレント -> パッケージ内 の順で探す。"""
    if not path or os.path.isabs(path):
        return path
    if caller_file and not caller_file.startswith("<"):
        candidate = os.path.join(os.path.dirname(os.path.abspath(caller_file)), path)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return resolve_model_path(path)


class YOLO(Detector):
    """ultralytics 風の書き方で使う検出器。

        from ketsugecam import YOLO

        model_yolo = YOLO("best.engine")
        dets = model_yolo(frame, conf=0.35, iou=0.45, max_det=5)

    相対パスは「YOLO() を呼んだスクリプトのフォルダ -> カレント -> ketsugecam/ 内」
    の順で探す。main.py の隣に置いたモデルがそのまま使える。

    predict() / __call__ の conf / iou / max_det / agnostic_nms はその呼び出しだけ
    有効 (ultralytics と同じ)。省略するとコンストラクタ (または属性) の値。
    返り値は Detection のリスト (ultralytics の Results ではない)。
    """

    def __init__(self, model: str = "best.engine",
                 conf: float = 0.25,
                 iou: float = 0.65,
                 max_det: int = 100,
                 agnostic_nms: bool = False) -> None:
        caller = sys._getframe(1).f_code.co_filename
        self.model_path = _resolve_from_caller(model, caller)
        super().__init__(self.model_path, conf=conf, iou=iou, max_det=max_det,
                         agnostic_nms=agnostic_nms)

    def predict(self, frame, conf: Optional[float] = None,
                iou: Optional[float] = None, max_det: Optional[int] = None,
                agnostic_nms: Optional[bool] = None) -> List[Detection]:
        y = self.yolo
        saved = (y.PROBABILITY_THRESHOLD, y.NMS_THRESHOLD, y.TOP_K, y.AGNOSTIC_NMS)
        if conf is not None:
            y.PROBABILITY_THRESHOLD = float(conf)
        if iou is not None:
            y.NMS_THRESHOLD = float(iou)
        if max_det is not None:
            y.TOP_K = int(max_det)
        if agnostic_nms is not None:
            y.AGNOSTIC_NMS = bool(agnostic_nms)
        try:
            return super().predict(frame)
        finally:
            (y.PROBABILITY_THRESHOLD, y.NMS_THRESHOLD, y.TOP_K,
             y.AGNOSTIC_NMS) = saved

    __call__ = predict
