# Dataset manifests

What the O1 and O2 results were computed from, so a later copy can be matched to it.

| Dataset | File | What it lists |
|---|---|---|
| MDB-stem-synth | (below) | The archive the team downloaded from Zenodo record 1481172: `MDB-stem-synth.tar.gz`, 1,845,117,819 bytes, MD5 `31c1f6b4888e5fd108af91c69789a809`. All 230 stems were used. |
| MIR-1K | `mir1k.sha256` | SHA-256 of every `Wavfile/*.wav` and `PitchLabel/*.pv` file of the team's copy (the mirlab.org original now returns 404). |
| VocalSet | `vocalset_pcm.md5` | MD5 of the 16-bit PCM samples of each of the 1,513 takes the O1 corpus draws from (VocalSet 1.1 `FULL/`, techniques straight, vibrato, slow_forte, slow_piano and messa), as `ffmpeg -i take.wav -f md5 -` prints it. Sample-level hashes do not depend on the WAV header, so they match VocalSet 1.2's `data_by_singer/` copies too. |

Check a copy with `sha256sum -c mir1k.sha256` from the folder above `MIR-1K/`.
