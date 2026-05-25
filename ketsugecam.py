"""
ketsugecam/ketsugecam.py

DXGI → CUDA 直接転送。CPUノータッチ。
返り値: torch.Tensor (CUDA, uint8, [H, W, 3], BGR)

フロー (CUDA interop有効時):
    DXGI Texture (GPU VRAM)
        ↓ CopySubresourceRegion  ROIのみ GPU→GPU
    D3D11 DEFAULT Texture (GPU VRAM)
        ↓ cudaGraphicsD3D11RegisterResource
        ↓ cudaGraphicsMapResources
        ↓ cudaMemcpy2DFromArray    GPU→GPU
    torch.Tensor (CUDA, BGRA)
        ↓ [..., :3] スライス (ゼロコピー)
    torch.Tensor (CUDA, BGR)  ← CPUノータッチ

フロー (CPUフォールバック):
    DXGI → Staging(Map) → torch.frombuffer → .cuda()
"""

import ctypes
import ctypes.wintypes as wintypes
import time
from threading import Thread, Event
from typing import Optional, Tuple

import comtypes
import torch

from .libs import (
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    D3D11_TEXTURE2D_DESC, D3D11_BOX, DXGI_SAMPLE_DESC,
    D3D11_USAGE_STAGING, D3D11_USAGE_DEFAULT,
    D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    IDXGIFactory1, IDXGIAdapter1, IDXGIOutput, IDXGIOutput1,
    IDXGIOutputDuplication, IDXGIResource, IDXGISurface,
    DXGI_OUTPUT_DESC, DXGI_OUTDUPL_FRAME_INFO, DXGI_MAPPED_RECT,
    DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_ERROR_NOT_FOUND,
)

# ─── CUDA Runtime (ctypes直接呼び出し) ───────────────────────────────────────
import os, glob as _glob

_CUDA_AVAILABLE = False
_cudart = None

def _load_cudart():
    # 既知パス優先
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
    ]
    # 自動探索
    for pattern in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\cudart64_*.dll",
        r"C:\Windows\System32\cudart64_*.dll",
    ]:
        candidates += _glob.glob(pattern)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None

_cudart = _load_cudart()
if _cudart is not None:
    # 使用する関数のシグネチャ設定
    # cudaGraphicsD3D11RegisterResource(resource*, pD3DResource, flags)
    _cudart.cudaGraphicsD3D11RegisterResource.restype  = ctypes.c_int
    _cudart.cudaGraphicsD3D11RegisterResource.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # cudaGraphicsResource_t*
        ctypes.c_void_p,                   # ID3D11Resource*
        ctypes.c_uint,                     # flags
    ]
    # cudaGraphicsMapResources(count, resources*, stream)
    _cudart.cudaGraphicsMapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsMapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGraphicsUnmapResources
    _cudart.cudaGraphicsUnmapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnmapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGraphicsResourceGetMappedMipmappedArray(array*, resource)
    _cudart.cudaGraphicsResourceGetMappedMipmappedArray.restype  = ctypes.c_int
    _cudart.cudaGraphicsResourceGetMappedMipmappedArray.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGetMipmappedArrayLevel(array*, mipmapped, level)
    _cudart.cudaGetMipmappedArrayLevel.restype  = ctypes.c_int
    _cudart.cudaGetMipmappedArrayLevel.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    # cudaMemcpy2DFromArray(dst, dpitch, src, woff, hoff, width, height, kind)
    _cudart.cudaMemcpy2DFromArray.restype  = ctypes.c_int
    _cudart.cudaMemcpy2DFromArray.argtypes = [
        ctypes.c_void_p,  # dst
        ctypes.c_size_t,  # dpitch
        ctypes.c_void_p,  # src cudaArray_t
        ctypes.c_size_t,  # wOffset
        ctypes.c_size_t,  # hOffset
        ctypes.c_size_t,  # width (bytes)
        ctypes.c_size_t,  # height
        ctypes.c_int,     # kind (3=DeviceToDevice)
    ]
    # cudaGraphicsUnregisterResource
    _cudart.cudaGraphicsUnregisterResource.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]

    _cudart.cudaSetDevice.restype  = ctypes.c_int
    _cudart.cudaSetDevice.argtypes = [ctypes.c_int]

    _CUDA_AVAILABLE = True

# cudaGraphicsRegisterFlagsReadOnly = 1
_CUDA_GRAPHICS_REGISTER_FLAGS_NONE     = 0  # ← これが正解
_CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

# ─── Windows 高精度タイマー ───────────────────────────────────────────────────
_kernel32 = ctypes.windll.kernel32
_INFINITE  = 0xFFFFFFFF
_WAIT_FAILED = 0xFFFFFFFF
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003
_DXGI_MAP_READ = 1


def _create_timer():
    h = _kernel32.CreateWaitableTimerExW(
        None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, _TIMER_ALL_ACCESS
    )
    if h == 0:
        raise ctypes.WinError()
    return h


def _set_timer(handle, target_fps: float):
    interval = ctypes.c_longlong(int(-10_000_000 / target_fps))
    if not _kernel32.SetWaitableTimer(handle, ctypes.byref(interval), 0, None, None, 0):
        raise ctypes.WinError()


def _wait_timer(handle):
    return _kernel32.WaitForSingleObject(handle, _INFINITE)


def _cancel_timer(handle):
    _kernel32.CancelWaitableTimer(handle)
    _kernel32.CloseHandle(handle)


# ─── DXGI/D3D 初期化ヘルパー ──────────────────────────────────────────────────
def _create_dxgi_factory():
    fn = ctypes.windll.dxgi.CreateDXGIFactory1
    fn.argtypes = (comtypes.GUID, ctypes.POINTER(ctypes.c_void_p))
    fn.restype  = ctypes.c_int32
    p = ctypes.c_void_p(0)
    fn(IDXGIFactory1._iid_, ctypes.byref(p))
    return ctypes.POINTER(IDXGIFactory1)(p.value)


def _enum_adapters(factory):
    adapters, i = [], 0
    while True:
        try:
            p = ctypes.POINTER(IDXGIAdapter1)()
            factory.EnumAdapters1(i, ctypes.byref(p))
            adapters.append(p)
            i += 1
        except comtypes.COMError as e:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == e.args[0]:
                break
            raise
    return adapters


def _enum_outputs(adapter):
    outputs, i = [], 0
    while True:
        try:
            p = ctypes.POINTER(IDXGIOutput)()
            adapter.EnumOutputs(i, ctypes.byref(p))
            outputs.append(p.QueryInterface(IDXGIOutput1))
            i += 1
        except comtypes.COMError as e:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == e.args[0]:
                break
            raise
    return outputs


def _create_d3d11_device(adapter):
    fn = ctypes.windll.d3d11.D3D11CreateDevice
    levels = (ctypes.c_uint * 3)(
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    )
    device  = ctypes.POINTER(ID3D11Device)()
    context = ctypes.POINTER(ID3D11DeviceContext)()
    fn(
        adapter, 0, None,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        ctypes.byref(levels), 3, 7,
        ctypes.byref(device), None, ctypes.byref(context),
    )
    im_ctx = ctypes.POINTER(ID3D11DeviceContext)()
    device.GetImmediateContext(ctypes.byref(im_ctx))
    return device, im_ctx


def _create_staging_texture(device, width, height):
    """CPU読み取り用 Staging テクスチャ (フォールバック用)"""
    desc = D3D11_TEXTURE2D_DESC()
    desc.Width          = width
    desc.Height         = height
    desc.MipLevels      = 1
    desc.ArraySize      = 1
    desc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM
    desc.SampleDesc     = DXGI_SAMPLE_DESC(1, 0)
    desc.Usage          = D3D11_USAGE_STAGING
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
    desc.BindFlags      = 0
    desc.MiscFlags      = 0
    tex = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), None, ctypes.byref(tex))
    surf = tex.QueryInterface(IDXGISurface)
    return tex, surf


def _create_default_texture(device, width, height):
    """CUDA interop用 DEFAULT テクスチャ (GPU VRAM常駐)"""
    D3D11_RESOURCE_MISC_SHARED = 0x2
    desc = D3D11_TEXTURE2D_DESC()
    desc.Width          = width
    desc.Height         = height
    desc.MipLevels      = 1
    desc.ArraySize      = 1
    desc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM
    desc.SampleDesc     = DXGI_SAMPLE_DESC(1, 0)
    desc.Usage          = D3D11_USAGE_DEFAULT
    desc.CPUAccessFlags = 0
    desc.BindFlags      = 0
    desc.MiscFlags      = D3D11_RESOURCE_MISC_SHARED
    tex = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), None, ctypes.byref(tex))
    return tex


# ─── KetsugeCam ───────────────────────────────────────────────────────────────
class KetsugeCam:
    """
    DXGI → CUDA 直接転送キャプチャ。CPUノータッチ。

    cuda-python が使える場合:
        GPU → GPU のみ。CPU転送ゼロ。
    フォールバック:
        GPU → CPU → CUDA (従来方式)
    """

    def __init__(
        self,
        output_idx: int = 0,
        device_idx: int = 0,
        region: Optional[Tuple[int, int, int, int]] = None,
        cuda: bool = True,
    ):
        # ── DXGI/D3D初期化 ── (torchより先にやる)
        factory  = _create_dxgi_factory()
        adapters = _enum_adapters(factory)
        adapter  = adapters[device_idx]
        self._adapter = adapter  # interop初期化で使う
        self._device, self._ctx = _create_d3d11_device(adapter)

        outputs  = _enum_outputs(adapter)
        p_out    = outputs[output_idx]

        out_desc = DXGI_OUTPUT_DESC()
        p_out.GetDesc(ctypes.byref(out_desc))
        self.width  = out_desc.DesktopCoordinates.right  - out_desc.DesktopCoordinates.left
        self.height = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top

        if region is None:
            region = (0, 0, self.width, self.height)
        self._validate_region(region)
        self.region = region

        roi_w = region[2] - region[0]
        roi_h = region[3] - region[1]
        self._roi_w = roi_w
        self._roi_h = roi_h

        # ── Duplicator ──
        self._dupl = ctypes.POINTER(IDXGIOutputDuplication)()
        p_out.DuplicateOutput(self._device, ctypes.byref(self._dupl))

        # ── D3D11_BOX キャッシュ ──
        self._roi_box        = D3D11_BOX()
        self._roi_box.left   = region[0]
        self._roi_box.top    = region[1]
        self._roi_box.front  = 0
        self._roi_box.right  = region[2]
        self._roi_box.bottom = region[3]
        self._roi_box.back   = 1

        self._frame_info = DXGI_OUTDUPL_FRAME_INFO()
        self._rect       = DXGI_MAPPED_RECT()

        # ── CUDA interop 初期化 ──
        # ★ torch.cuda.is_available() より先に呼ぶ
        # PyTorchがCUDAコンテキストを作る前にD3D11と紐付ける必要がある
        self._interop  = False
        self._cuda_res = None
        self._gpu_buf  = None
        self._gpu_ptr  = None
        self._stage_tex  = None
        self._stage_surf = None

        if cuda and _CUDA_AVAILABLE:
            self._init_cuda_interop(roi_w, roi_h)

        # interop後にtorchのCUDA有効確認
        self.cuda = cuda and torch.cuda.is_available()

        # interop失敗 or 非CUDA → Stagingフォールバック
        if not self._interop:
            self._stage_tex, self._stage_surf = _create_staging_texture(
                self._device, roi_w, roi_h
            )

        # ── 連続キャプチャ用 ──
        self._latest: Optional[torch.Tensor] = None
        self._frame_event = Event()
        self._stop_event  = Event()
        self._thread: Optional[Thread] = None
        self.is_capturing  = False
        self._capture_fps: float = 0.0

    def _init_cuda_interop(self, roi_w, roi_h):
        try:
            # ── Driver API (nvcuda.dll) でD3D11と紐付けたCUDAコンテキスト作成 ──
            _cuda_driver = ctypes.CDLL(r'C:\Windows\System32\nvcuda.dll')

            _cuda_driver.cuInit.restype  = ctypes.c_int
            _cuda_driver.cuInit.argtypes = [ctypes.c_uint]
            err = _cuda_driver.cuInit(0)
            if err != 0:
                raise RuntimeError(f"cuInit failed: {err}")

            _cuda_driver.cuD3D11GetDevice.restype  = ctypes.c_int
            _cuda_driver.cuD3D11GetDevice.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
            cu_dev      = ctypes.c_int(-1)
            adapter_ptr = ctypes.cast(self._adapter, ctypes.c_void_p).value
            err = _cuda_driver.cuD3D11GetDevice(ctypes.byref(cu_dev), ctypes.c_void_p(adapter_ptr))
            if err != 0:
                raise RuntimeError(f"cuD3D11GetDevice failed: {err}")
            print(f"[ketsugecam] CUDA device: {cu_dev.value}")

            # D3D11デバイスと紐付けたCUDAコンテキスト作成
            _cuda_driver.cuD3D11CtxCreate_v2.restype  = ctypes.c_int
            _cuda_driver.cuD3D11CtxCreate_v2.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                ctypes.c_uint, ctypes.c_void_p,
            ]
            d3d_ptr = ctypes.cast(self._device, ctypes.c_void_p).value
            cu_ctx  = ctypes.c_void_p(0)
            err = _cuda_driver.cuD3D11CtxCreate_v2(
                ctypes.byref(cu_ctx), cu_dev.value, 0, ctypes.c_void_p(d3d_ptr)
            )
            if err != 0:
                raise RuntimeError(f"cuD3D11CtxCreate_v2 failed: {err}")

            # コンテキストをカレントに設定
            _cuda_driver.cuCtxSetCurrent.restype  = ctypes.c_int
            _cuda_driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            err = _cuda_driver.cuCtxSetCurrent(cu_ctx.value)
            if err != 0:
                raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

            # CUDA interop用テクスチャ作成
            self._interop_tex = _create_default_texture(self._device, roi_w, roi_h)
            # ★ p1方式: ctypes.cast(tex, c_void_p).value が正しいポインタ
            tex_ptr = ctypes.cast(self._interop_tex, ctypes.c_void_p).value
            if not tex_ptr:
                raise RuntimeError("テクスチャ作成失敗 (NULL)")
            print(f"[ketsugecam] tex_ptr: {hex(tex_ptr)}")

            # ★ torch.emptyを先に呼んでPyTorchのCUDAコンテキストを確立
            # これをRegisterResourceより後にやるとコンテキストが変わって400になる
            self._gpu_buf = torch.empty(
                (roi_h, roi_w, 4), dtype=torch.uint8, device="cuda"
            )
            self._gpu_ptr = self._gpu_buf.data_ptr()

            # コンテキストをD3D11のものに戻す
            err = _cuda_driver.cuCtxSetCurrent(cu_ctx.value)
            if err != 0:
                raise RuntimeError(f"cuCtxSetCurrent(restore) failed: {err}")

            # ★ Driver APIで登録
            _cuda_driver.cuGraphicsD3D11RegisterResource.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsD3D11RegisterResource.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            self._cuda_res = ctypes.c_void_p(0)
            err = _cuda_driver.cuGraphicsD3D11RegisterResource(
                ctypes.byref(self._cuda_res),
                ctypes.c_void_p(tex_ptr),
                _CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
            )
            print(f"[ketsugecam] cuGraphicsD3D11RegisterResource: err={err} res={self._cuda_res.value}")
            if err != 0:
                raise RuntimeError(f"cuGraphicsD3D11RegisterResource failed: {err}")

            # Driver API関数シグネチャ設定
            _cuda_driver.cuCtxSetCurrent.restype  = ctypes.c_int
            _cuda_driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            _cuda_driver.cuGraphicsMapResources.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsMapResources.argtypes = [
                ctypes.c_uint,      # count
                ctypes.c_void_p,    # resources* (配列の先頭アドレス)
                ctypes.c_void_p,    # stream
            ]
            _cuda_driver.cuGraphicsUnmapResources.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsUnmapResources.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _cuda_driver.cuGraphicsResourceGetMappedMipmappedArray.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsResourceGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
            _cuda_driver.cuMipmappedArrayGetLevel.restype  = ctypes.c_int
            _cuda_driver.cuMipmappedArrayGetLevel.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            _cuda_driver.cuMemcpy2D_v2.restype  = ctypes.c_int
            _cuda_driver.cuMemcpy2D_v2.argtypes = [ctypes.c_void_p]

            self._cu_ctx    = cu_ctx
            self._cu_driver = _cuda_driver
            self._interop  = True
            print("[ketsugecam] モード: CUDA interop (CPUノータッチ)")

        except Exception as e:
            print(f"[ketsugecam] CUDA interop 失敗 → CPUフォールバック: {e}")
            self._interop = False

    # ── 単発キャプチャ ────────────────────────────────────────────────────────
    def grab(self) -> Optional[torch.Tensor]:
        return self._grab()

    def _grab(self) -> Optional[torch.Tensor]:
        res_ptr = ctypes.POINTER(IDXGIResource)()
        try:
            self._dupl.AcquireNextFrame(
                1,
                ctypes.byref(self._frame_info),
                ctypes.byref(res_ptr),
            )
        except comtypes.COMError as e:
            code = ctypes.c_int32(e.args[0]).value
            if code == ctypes.c_int32(DXGI_ERROR_WAIT_TIMEOUT).value:
                return None
            if code == ctypes.c_int32(DXGI_ERROR_ACCESS_LOST).value:
                self._rebuild_dupl()
                return None
            raise

        try:
            src_tex = res_ptr.QueryInterface(ID3D11Texture2D)
            dst_tex = self._interop_tex if self._interop else self._stage_tex
            self._ctx.CopySubresourceRegion(
                dst_tex, 0, 0, 0, 0,
                src_tex, 0,
                ctypes.byref(self._roi_box),
            )
        finally:
            self._dupl.ReleaseFrame()

        if self._interop:
            return self._to_tensor_cuda()
        else:
            self._stage_surf.Map(ctypes.byref(self._rect), _DXGI_MAP_READ)
            try:
                frame = self._to_tensor_cpu()
            finally:
                self._stage_surf.Unmap()
            return frame

    def _to_tensor_cuda(self) -> torch.Tensor:
        """GPU → GPU 直接転送。CPUノータッチ。Driver API統一。"""
        cu = self._cu_driver
        # 毎フレーム D3D11コンテキストを確実にカレントに設定
        cu.cuCtxSetCurrent(self._cu_ctx.value)

        res_val = self._cuda_res.value
        res_arr = (ctypes.c_void_p * 1)(res_val)

        err = cu.cuGraphicsMapResources(1, ctypes.addressof(res_arr), None)
        if err != 0:
            raise RuntimeError(f"cuGraphicsMapResources failed: {err}")
        try:
            # cuGraphicsResourceGetMappedMipmappedArray
            mip_arr = ctypes.c_void_p(0)
            err = cu.cuGraphicsResourceGetMappedMipmappedArray(
                ctypes.byref(mip_arr), self._cuda_res
            )
            if err != 0:
                raise RuntimeError(f"GetMappedMipmappedArray failed: {err}")

            # cuMipmappedArrayGetLevel
            cuda_arr = ctypes.c_void_p(0)
            err = cu.cuMipmappedArrayGetLevel(
                ctypes.byref(cuda_arr), mip_arr, 0
            )
            if err != 0:
                raise RuntimeError(f"cuMipmappedArrayGetLevel failed: {err}")

            # cuMemcpy2D: cudaArray → GPU buffer
            # CUDA_MEMCPY2D 構造体
            class CUDA_MEMCPY2D(ctypes.Structure):
                _fields_ = [
                    ("srcXInBytes",   ctypes.c_size_t),
                    ("srcY",          ctypes.c_size_t),
                    ("srcMemoryType", ctypes.c_int),   # CU_MEMORYTYPE_ARRAY=3
                    ("srcHost",       ctypes.c_void_p),
                    ("srcDevice",     ctypes.c_uint64),
                    ("srcArray",      ctypes.c_void_p),
                    ("srcPitch",      ctypes.c_size_t),
                    ("dstXInBytes",   ctypes.c_size_t),
                    ("dstY",          ctypes.c_size_t),
                    ("dstMemoryType", ctypes.c_int),   # CU_MEMORYTYPE_DEVICE=2
                    ("dstHost",       ctypes.c_void_p),
                    ("dstDevice",     ctypes.c_uint64),
                    ("dstArray",      ctypes.c_void_p),
                    ("dstPitch",      ctypes.c_size_t),
                    ("WidthInBytes",  ctypes.c_size_t),
                    ("Height",        ctypes.c_size_t),
                ]

            cp = CUDA_MEMCPY2D()
            cp.srcXInBytes   = 0
            cp.srcY          = 0
            cp.srcMemoryType = 3          # CU_MEMORYTYPE_ARRAY
            cp.srcHost       = None
            cp.srcDevice     = 0
            cp.srcArray      = cuda_arr
            cp.srcPitch      = 0
            cp.dstXInBytes   = 0
            cp.dstY          = 0
            cp.dstMemoryType = 2          # CU_MEMORYTYPE_DEVICE
            cp.dstHost       = None
            cp.dstDevice     = self._gpu_ptr
            cp.dstArray      = None
            cp.dstPitch      = self._roi_w * 4
            cp.WidthInBytes  = self._roi_w * 4
            cp.Height        = self._roi_h

            err = cu.cuMemcpy2D_v2(ctypes.byref(cp))
            if err != 0:
                raise RuntimeError(f"cuMemcpy2D_v2 failed: {err}")

        finally:
            cu.cuGraphicsUnmapResources(1, ctypes.addressof(res_arr), None)

        # BGRA → BGR スライス (ゼロコピー)
        return self._gpu_buf[..., :3]

    def _to_tensor_cpu(self) -> torch.Tensor:
        """CPUフォールバック: Staging Map → torch → .cuda()"""
        pitch  = int(self._rect.Pitch)
        stride = pitch // 4
        size   = pitch * self._roi_h
        buf = (ctypes.c_uint8 * size).from_address(
            ctypes.addressof(self._rect.pBits.contents)
        )
        t = torch.frombuffer(buf, dtype=torch.uint8).reshape(self._roi_h, stride, 4)
        if stride != self._roi_w:
            t = t[:, :self._roi_w, :]
        t = t[..., :3]
        if self.cuda:
            t = t.to("cuda", non_blocking=True)
        return t

    # ── 連続キャプチャ ────────────────────────────────────────────────────────
    def start(self, target_fps: float = 60.0):
        if self.is_capturing:
            return
        self.is_capturing = True
        self._stop_event.clear()
        self._frame_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            args=(target_fps,),
            name="KetsugeCam",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        if not self.is_capturing:
            return
        self._stop_event.set()
        self._frame_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_capturing = False
        self._latest = None
        self._frame_event.clear()
        self._stop_event.clear()

    def get_latest_frame(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        self._frame_event.wait(timeout=timeout)
        self._frame_event.clear()
        return self._latest

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def mode(self) -> str:
        return "cuda_interop" if self._interop else "cpu_fallback"

    def _capture_loop(self, target_fps: float):
        if self._interop:
            self._cu_driver.cuCtxSetCurrent(self._cu_ctx.value)

        timer   = _create_timer()
        _set_timer(timer, target_fps)
        count   = 0
        t_start = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if _wait_timer(timer) == _WAIT_FAILED:
                    break
                _set_timer(timer, target_fps)
                frame = self._grab()
                if frame is not None:
                    self._latest = frame
                    self._frame_event.set()
                    count += 1
                elapsed = time.perf_counter() - t_start
                if elapsed >= 0.5:
                    self._capture_fps = count / elapsed
                    count   = 0
                    t_start = time.perf_counter()
        finally:
            _cancel_timer(timer)

    # ── Duplicator 再構築 ─────────────────────────────────────────────────────
    def _rebuild_dupl(self):
        time.sleep(0.1)
        try:
            self._dupl.ReleaseFrame()
        except Exception:
            pass
        try:
            self._dupl.Release()
        except Exception:
            pass
        factory  = _create_dxgi_factory()
        adapters = _enum_adapters(factory)
        outputs  = _enum_outputs(adapters[0])
        self._dupl = ctypes.POINTER(IDXGIOutputDuplication)()
        outputs[0].DuplicateOutput(self._device, ctypes.byref(self._dupl))

    # ── ユーティリティ ────────────────────────────────────────────────────────
    def _validate_region(self, region):
        l, t, r, b = region
        if not (self.width >= r > l >= 0 and self.height >= b > t >= 0):
            raise ValueError(f"Invalid region {region} for {self.width}x{self.height}")

    def release(self):
        self.stop()
        if self._cuda_res is not None:
            try:
                _cudart.cudaGraphicsUnregisterResource(self._cuda_res)
            except Exception:
                pass
        if self._stage_surf is not None:
            try:
                self._stage_surf.Unmap()
            except Exception:
                pass
        if self._stage_tex is not None:
            try:
                self._stage_tex.Release()
            except Exception:
                pass
        try:
            self._dupl.ReleaseFrame()
            self._dupl.Release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    def __repr__(self):
        return (
            f"<KetsugeCam {self.width}x{self.height} "
            f"region={self.region} mode={self.mode}>"
        )