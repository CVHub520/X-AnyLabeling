import json
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TrtBaseModel:

    def __init__(
        self,
        model_path: str,
        device_type: str = "gpu",
        log_severity_level: int = 3,
    ) -> None:
        try:
            import tensorrt as trt  # type: ignore
            from cuda.bindings import runtime as cudart  # type: ignore
        except ImportError as e:
            raise ImportError(
                "TensorRT execution provider requires the 'tensorrt' and "
                "'cuda-python' packages.\nInstall them with: 'uv pip install tensorrt cuda-python'"
            ) from e

        self._trt = trt
        self._cudart = cudart
        self.model_path = model_path
        self.device_type = device_type

        severity_map = {
            0: trt.Logger.VERBOSE,
            1: trt.Logger.INFO,
            2: trt.Logger.WARNING,
            3: trt.Logger.ERROR,
            4: trt.Logger.INTERNAL_ERROR,
        }
        self.logger = trt.Logger(
            severity_map.get(log_severity_level, trt.Logger.ERROR)
        )
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        self.metadata: Dict[str, Any] = {}
        with open(model_path, "rb") as f:
            engine_bytes = self._strip_ultralytics_prefix(f.read())

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {model_path}"
            )
        self.context = self.engine.create_execution_context()

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        self.tensor_dtypes: Dict[str, np.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self._trt_dtype_to_np(self.engine.get_tensor_dtype(name))
            self.tensor_dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape

        self.device_inputs: Dict[str, int] = {}
        self.device_outputs: Dict[str, int] = {}
        self.host_outputs: Dict[str, np.ndarray] = {}

        err, self.stream = cudart.cudaStreamCreate()
        self._check(err)

        self._allocate_static_buffers()

    def get_input_name(self) -> str:
        return self.input_names[0]

    def get_input_shape(self) -> List[int]:
        return list(self.input_shapes[self.input_names[0]])

    def get_output_name(self) -> List[str]:
        return list(self.output_names)

    def get_metadata_info(self, field: str) -> Optional[str]:
        if field in self.metadata:
            value = self.metadata[field]
            return value if isinstance(value, str) else json.dumps(value)
        return None

    def get_ort_inference(
        self,
        blob: Optional[np.ndarray] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        extract: bool = True,
        squeeze: bool = False,
    ):
        if inputs is None:
            inputs = {self.get_input_name(): blob}
        outs_dict = self._infer(inputs)
        outs = [outs_dict[name] for name in self.output_names]
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs

    def _strip_ultralytics_prefix(self, data: bytes) -> bytes:
        if len(data) < 4:
            return data
        meta_len = struct.unpack("<I", data[:4])[0]
        if 0 < meta_len < len(data) - 4:
            try:
                meta = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return data
            if isinstance(meta, dict):
                self.metadata = meta
                return data[4 + meta_len :]
        return data

    def _trt_dtype_to_np(self, dtype) -> np.dtype:
        trt = self._trt
        mapping = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int8: np.int8,
            trt.bool: np.bool_,
        }
        for attr, np_type in (("int64", np.int64), ("uint8", np.uint8)):
            if hasattr(trt, attr):
                mapping[getattr(trt, attr)] = np_type
        return np.dtype(mapping.get(dtype, np.float32))

    def _check(self, err) -> None:
        if int(err) != int(self._cudart.cudaError_t.cudaSuccess):
            raise RuntimeError(f"CUDA runtime error: {err}")

    def _alloc_device(self, nbytes: int) -> int:
        err, ptr = self._cudart.cudaMalloc(nbytes)
        self._check(err)
        return int(ptr)

    def _free_device(self, ptr: int) -> None:
        if ptr:
            self._cudart.cudaFree(ptr)

    def _allocate_static_buffers(self) -> None:
        for name in self.input_names:
            shape = self.input_shapes[name]
            if any(s < 0 for s in shape):
                continue
            nbytes = int(np.prod(shape)) * self.tensor_dtypes[name].itemsize
            self.device_inputs[name] = self._alloc_device(nbytes)
        for name in self.output_names:
            shape = self.output_shapes[name]
            if any(s < 0 for s in shape):
                continue
            self._ensure_output_buffer(name, shape)

    def _ensure_output_buffer(self, name: str, shape: Tuple[int, ...]) -> None:
        dtype = self.tensor_dtypes[name]
        nbytes = int(np.prod(shape)) * dtype.itemsize
        host = self.host_outputs.get(name)
        if host is None or host.shape != shape or host.dtype != dtype:
            self._free_device(self.device_outputs.pop(name, 0))
            self.device_outputs[name] = self._alloc_device(nbytes)
            self.host_outputs[name] = np.empty(shape, dtype=dtype)
        self.output_shapes[name] = shape

    def _infer(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cudart = self._cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        for name in self.input_names:
            if name not in feed:
                raise KeyError(f"Missing TensorRT input: {name}")
            arr = np.ascontiguousarray(
                feed[name].astype(self.tensor_dtypes[name], copy=False)
            )
            shape = arr.shape
            if shape != self.input_shapes[name]:
                self.context.set_input_shape(name, shape)
                self._free_device(self.device_inputs.pop(name, 0))
                self.device_inputs[name] = self._alloc_device(arr.nbytes)
                self.input_shapes[name] = shape
            elif name not in self.device_inputs:
                self.device_inputs[name] = self._alloc_device(arr.nbytes)

            (err,) = cudart.cudaMemcpyAsync(
                self.device_inputs[name],
                arr.ctypes.data,
                arr.nbytes,
                H2D,
                self.stream,
            )
            self._check(err)
            self.context.set_tensor_address(
                name, int(self.device_inputs[name])
            )

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(s < 0 for s in shape):
                raise RuntimeError(
                    f"Unresolved dynamic output shape for tensor '{name}'."
                )
            self._ensure_output_buffer(name, shape)
            self.context.set_tensor_address(
                name, int(self.device_outputs[name])
            )

        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        outputs: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            host = self.host_outputs[name]
            (err,) = cudart.cudaMemcpyAsync(
                host.ctypes.data,
                self.device_outputs[name],
                host.nbytes,
                D2H,
                self.stream,
            )
            self._check(err)
            outputs[name] = host

        (err,) = cudart.cudaStreamSynchronize(self.stream)
        self._check(err)
        return outputs

    def __del__(self) -> None:
        try:
            cudart = self._cudart
            for ptr in self.device_inputs.values():
                cudart.cudaFree(ptr)
            for ptr in self.device_outputs.values():
                cudart.cudaFree(ptr)
            if getattr(self, "stream", None) is not None:
                cudart.cudaStreamDestroy(self.stream)
        except Exception:
            pass
