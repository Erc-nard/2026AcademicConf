"""Strict video-only OLKAVS checkpoint adapter for a CPU smoke test.

CTC greedy and upstream hybrid decoding are available, with prototype crops.
No reference transcript or audio is accepted by this API.
"""

import csv
import subprocess
import sys
from pathlib import Path

from .data import write_jsonl
from .preprocessing import sha256


def collapse_ctc(frame_ids, blank=3, eos=2):
    tokens = []
    previous = None
    for token in frame_ids:
        if token != previous and token != blank:
            if token == eos:
                break
            if token not in {0, 1}:
                tokens.append(token)
        previous = token
    return tokens


def infer_olkavs(mouth, checkpoint, repo, sample_id, output, threads=2, decoder="ctc_greedy"):
    import numpy as np
    import torch
    if threads < 1:
        raise ValueError("threads must be positive")
    if decoder not in {"ctc_greedy", "hybrid"}:
        raise ValueError("Unknown decoder")
    repo = Path(repo).resolve()
    if not (repo / "avsr/utils/model_builder.py").is_file():
        raise ValueError("Clone the official OLKAVS repository first")
    revision = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
                               "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(repo))
    from avsr.utils.model_builder import build_model
    from vocabulary.utils import grp2char
    with (repo / "vocabulary/kor_grapheme.csv").open(encoding="utf-8-sig", newline="") as stream:
        vocabulary = {int(row["id"]): row["char"] for row in csv.DictReader(stream)}
    if len(vocabulary) != 72 or vocabulary.get(3) != "<unk>":
        raise ValueError("Unexpected OLKAVS vocabulary")
    with np.load(mouth, allow_pickle=False) as archive:
        frames = archive["frames"]
        fps = float(archive["fps"])
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
            or not 1 <= len(frames) <= 750 or frames.shape[1:3] != (96, 96)):
        raise ValueError("Expected 1..750 uint8 BGR mouth frames of size 96x96")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid input FPS")
    torch.set_num_threads(threads)
    model = build_model(
        72, 0, architecture="video", loss_fn="hybrid", front_dim=512,
        encoder_n_layer=12, encoder_d_model=256, encoder_n_head=8,
        encoder_ff_dim=2048, encoder_dropout_p=.1,
        decoder_n_layer=6, decoder_d_model=256, decoder_n_head=8,
        decoder_ff_dim=2048, decoder_dropout_p=.1,
        pass_visual_frontend=False, verbose=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    video = torch.from_numpy(frames).float()
    std = video.std()
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Input video has zero or invalid variance")
    video = ((video - video.mean()) / std).permute(3, 0, 1, 2).unsqueeze(0)
    lengths = torch.tensor([len(frames)], dtype=torch.long)
    with torch.inference_mode():
        if decoder == "ctc_greedy":
            encoded = model.encoder(video, lengths)  # No audio arguments.
            features = model.medium(encoded)
            logits = model.decoder.ctcdecoder(features)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits")
            tokens = collapse_ctc(logits.argmax(-1).tolist())
        else:
            from avsr.utils.search import SearchSequence
            search = SearchSequence(model, 72, method="hybrid", max_len=200, ctc_rate=.6, mp_num=1)
            output_ids, output_lengths = search(video, lengths, None, None, beam_size=1,
                                                D_end=-14, M_end=16, device="cpu", mp_num=1)
            tokens = []
            for token in output_ids[0, :int(output_lengths[0])].tolist():
                if token == 2:
                    break
                if token not in {0, 1, 3}:
                    tokens.append(token)
    graphemes = "".join(vocabulary[token] for token in tokens)
    row = dict(id=sample_id, hypothesis=grp2char(graphemes), raw_graphemes=graphemes,
               token_ids=tokens, decoder=decoder, confidence=None,
               model="OLKAVS multiview_lip", checkpoint_sha256=sha256(checkpoint),
               upstream_commit=revision, torch_version=torch.__version__,
               preprocessing="mediapipe-mouth-prototype-v1; BGR global mean/std; no FPS conversion",
               vocabulary_sha256=sha256(repo / "vocabulary/kor_grapheme.csv"),
               input_sha256=sha256(mouth), input_fps=fps, input_frames=len(frames),
               audio_used=False, reference_used=False, device="cpu",
               status="cross-dataset preprocessing smoke test; not paper reproduction")
    write_jsonl(output, [row])
    return row
