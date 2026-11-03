"""Export a pitch tracker (preprocessing + network + decoding) to ONNX for the deployment layer
(proposal 5.5), then check it against the PyTorch model.

    python scripts/export_onnx.py --model fcpe  --out models/fcpe.onnx    # the tracker O2 chose
    python scripts/export_onnx.py --model pesto --out models/pesto.onnx --step-ms 10

The app picks the models up from ./models, ~/.adaptitune or next to the packaged executable (see
``find_onnx_model``), so it runs FCPE with ONNX Runtime and needs no PyTorch. Each model takes a
float32 waveform [1, n] at 16 kHz and returns (f0_hz[1, T], confidence[1, T]) every 10 ms.
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


class FcpeWrapper(torch.nn.Module):
    """FCPE as ``FcpeTracker`` runs it: torchfcpe's mel front end (16 kHz, 128 bands, 1024-point
    frames every 160 samples), the network, and its local-argmax decoder without the voicing
    mask. Written out with operations ONNX can export; ``confidence`` is the peak latent."""

    def __init__(self, infer):
        super().__init__()
        mel = infer.wav2mel.mel_extractor
        self.net = infer.model
        self.register_buffer("mel_basis", mel.mel_basis.clone())
        self.register_buffer("window", torch.hann_window(mel.win_size))
        self.register_buffer("cent_table", infer.model.cent_table.clone())
        self.n_fft, self.win, self.hop, self.clip = mel.n_fft, mel.win_size, mel.hop_length, mel.clip_val

    def forward(self, x):
        n = x.shape[-1]
        pad_l, pad_r = (self.win - self.hop) // 2, (self.win - self.hop + 1) // 2
        y = torch.nn.functional.pad(x.unsqueeze(1), (pad_l, pad_r), mode="reflect").squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=False, normalized=False, onesided=True, return_complex=False)
        mag = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-9)
        mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, mag), min=self.clip)).transpose(-1, -2)
        mel = torch.cat((mel, mel[:, -1:, :]), 1)[:, : n // self.hop + 1, :]  # torchfcpe's frame count
        latent = self.net(mel)  # (1, T, 360)
        conf, idx = latent.max(dim=-1, keepdim=True)
        local = ((idx - 4) + torch.arange(9)).clamp(0, latent.shape[-1] - 1)
        y_l = torch.gather(latent, -1, local)
        cents = (self.cent_table[local] * y_l).sum(-1) / y_l.sum(-1)
        return 10.0 * 2 ** (cents / 1200.0), conf.squeeze(-1)


def _export(w, out: Path, sr: int):
    x = torch.randn(1, sr * 3) * 0.1
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        w, (x,), str(out), input_names=["audio"], output_names=["f0", "confidence"],
        dynamic_axes={"audio": {1: "samples"}, "f0": {1: "frames"}, "confidence": {1: "frames"}},
        opset_version=17, dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pesto", "fcpe"], default="pesto")
    ap.add_argument("--out", help="default: models/<model>.onnx")
    ap.add_argument("--step-ms", type=float, default=10.0, help="PESTO only; FCPE's step is fixed at 10 ms")
    ap.add_argument("--checkpoint", default="mir-1k_g7", help="PESTO only")
    a = ap.parse_args()
    sr = 16_000
    out = Path(a.out or f"models/{a.model}.onnx")
    if a.model == "fcpe":
        import contextlib
        import sys

        from torchfcpe import spawn_bundled_infer_model

        with contextlib.redirect_stdout(sys.stderr):
            w = FcpeWrapper(spawn_bundled_infer_model(device="cpu")).eval()
        _export(w, out, sr)
        _check(w, out, sr)
        return
    import pesto

    _onnx_friendly_preprocessor()
    model = pesto.load_model(a.checkpoint, step_size=a.step_ms, sampling_rate=sr).eval()
    w = Wrapper(model).eval()
    _export(w, out, sr)
    _check(w, out, sr)


def _check(w, out: Path, sr: int):
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
