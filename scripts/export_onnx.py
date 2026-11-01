"""Export PESTO (preprocessing + network + decoding) to ONNX for the deployment layer
(proposal 5.5), then check it against the PyTorch model.

    python scripts/export_onnx.py --out models/pesto.onnx --step-ms 10

The app picks it up automatically from ./models/pesto.onnx, ~/.adaptitune/pesto.onnx or
$ADAPTITUNE_ONNX (see ``find_onnx_model``); or pass ``Corrector(tracker="models/pesto.onnx")``. The ONNX
model takes a float32 waveform [1, n] at 16 kHz and returns (f0_hz[1, T], confidence[1, T]).
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred, conf, _vol, _act = self.model(x, convert_to_freq=True, return_activations=True)
        return pred, conf


def _onnx_friendly_preprocessor():
    """PESTO's preprocessor builds a complex tensor (view_as_complex), which ONNX cannot
    export. Compute the same log-magnitude from the real and imaginary parts instead."""
    from pesto.data import Preprocessor

    eps = torch.finfo(torch.float32).eps

    def forward(self, x, sr=None):
        h = self.hcqt(x, sr=sr)  # (batch, harmonics, freqs, time, 2)
        mag = torch.sqrt(h[..., 0] ** 2 + h[..., 1] ** 2).permute(0, 3, 1, 2)
        return 20 * torch.log10(torch.clamp(mag, min=eps))

    Preprocessor.forward = forward


def main():
    import pesto

    _onnx_friendly_preprocessor()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/pesto.onnx")
    ap.add_argument("--step-ms", type=float, default=10.0)
    ap.add_argument("--checkpoint", default="mir-1k_g7")
    a = ap.parse_args()
    sr = 16_000
    model = pesto.load_model(a.checkpoint, step_size=a.step_ms, sampling_rate=sr).eval()
    w = Wrapper(model).eval()
    x = torch.randn(1, sr * 3) * 0.1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        w, (x,), str(out), input_names=["audio"], output_names=["f0", "confidence"],
        dynamic_axes={"audio": {1: "samples"}, "f0": {1: "frames"}, "confidence": {1: "frames"}},
        opset_version=17, dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    t = np.arange(sr * 2) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None]
    f_onnx, _ = sess.run(None, {"audio": y})
    with torch.no_grad():
        f_torch, _ = w(torch.from_numpy(y))
    diff = np.nanmedian(np.abs(1200 * np.log2(f_onnx / f_torch.numpy())))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB); median |onnx - torch| = {diff:.3f} cents; "
          f"median f0 on a 220 Hz tone = {np.median(f_onnx):.1f} Hz")


if __name__ == "__main__":
    main()
