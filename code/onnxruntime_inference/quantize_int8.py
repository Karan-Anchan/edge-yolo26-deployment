"""Quantize the FP32 ONNX to INT8 (QDQ) for CPU deployment via ONNX Runtime.

This is the CPU runtime's answer to the TensorRT INT8 rung — same idea (8-bit),
different engine (ONNX Runtime on the Ryzen 7700, no GPU).

Key choice — **per_channel=True**: the TensorRT INT8 experiment showed this
model's accuracy loss under INT8 is a *granularity* problem (localization boxes
go coarse), and FP8 confirmed 8-ish bits are plenty if spent well. Per-channel
weight quantization gives each output channel its own scale instead of one scale
for the whole tensor — the standard fix for exactly that symptom. So we give ORT
INT8 the best shot from the start rather than repeating the TensorRT INT8 cliff.

    fp32.onnx --(quant_pre_process)--> prepped.onnx
              --(quantize_static, QDQ, per-channel, 120 calib imgs)--> int8.onnx

QDQ = the graph carries explicit QuantizeLinear/DequantizeLinear pairs, which is
the format ORT's CPU kernels (and later onnxruntime-web) execute most efficiently.
"""

from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

# The detection head is the `/model.23/*` block (Detect module): box decode,
# the Sigmoid confidence branch, TopK, GatherElements. Quantizing it squashes
# all confidences to 0 (the model emits nothing). Standard detection-quant
# practice: keep the head in FP32, quantize only the backbone/neck (the bulk of
# the FLOPs, so we keep most of the speed + size win) — and the head is exactly
# where box/confidence precision matters most.
HEAD_PREFIX = "/model.23/"

REPO_ROOT = Path(__file__).resolve().parents[2]
FP32_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp32.onnx"
PREP_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_prepped.onnx"
INT8_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_int8_ort.onnx"
CALIB_NPY = REPO_ROOT / "models" / "onnx" / "fp4_calib.npy"
INPUT_NAME = "images"


class NpyCalibReader(CalibrationDataReader):
    """Feeds the 120 preprocessed calibration images one at a time."""

    def __init__(self, npy: Path, input_name: str):
        self.data = np.load(npy)
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        x = self.data[self.i:self.i + 1]  # (1,3,640,640) float32
        self.i += 1
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def main() -> None:
    # skip_symbolic_shape=True: the NMS-free head's TopK node (top-600 selection)
    # breaks ORT's symbolic shape inference. The graph is already onnxslim-
    # simplified at export, so we skip that pass and keep the rest of pre-process.
    print("[ort-int8] pre-process pass (symbolic shape inference skipped for TopK)...")
    quant_pre_process(str(FP32_ONNX), str(PREP_ONNX), skip_symbolic_shape=True)

    prepped = onnx.load(str(PREP_ONNX))
    head_nodes = [n.name for n in prepped.graph.node if HEAD_PREFIX in n.name]
    print(f"[ort-int8] excluding {len(head_nodes)} detection-head nodes from quantization")

    print("[ort-int8] static QDQ quantization (per-channel, backbone only, 120 calib imgs)...")
    quantize_static(
        model_input=str(PREP_ONNX),
        model_output=str(INT8_ONNX),
        calibration_data_reader=NpyCalibReader(CALIB_NPY, INPUT_NAME),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=head_nodes,
    )
    size = INT8_ONNX.stat().st_size / 1e6
    fp32_size = FP32_ONNX.stat().st_size / 1e6
    print(f"[ort-int8] wrote {INT8_ONNX.name} ({size:.1f} MB, "
          f"{fp32_size/size:.1f}x smaller than FP32's {fp32_size:.1f} MB)")


if __name__ == "__main__":
    main()
