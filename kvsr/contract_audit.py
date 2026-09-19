"""Reproducible, transcript-free comparison with the installed official OLKAVS code.

Run only against a reviewed upstream checkout: extracting a function is not a
security sandbox. No source video, transcript or predicted text is exported.
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

from .preprocessing import sha256


def run(repo, checkpoint, source, output):
    import cv2
    import numpy as np
    import torch
    from .missing import normalize_available
    from .olkavs import chunk_frontend, collapse_ctc, prepare_frames

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    from avsr.utils.model_builder import build_model
    from vocabulary.utils import char2grp, grp2char

    # Execute the reviewed function verbatim without unrelated audio imports.
    dataset = repo / 'dataset/dataset.py'
    tree = ast.parse(dataset.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_parse_video')
    namespace = dict(os=os, np=np, torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(dataset), 'exec'), namespace)
    model = build_model(72, 0, architecture='video', loss_fn='hybrid', front_dim=512,
        encoder_n_layer=12, encoder_d_model=256, encoder_n_head=8, encoder_ff_dim=2048,
        encoder_dropout_p=.1, decoder_n_layer=6, decoder_d_model=256, decoder_n_head=8,
        decoder_ff_dim=2048, decoder_dropout_p=.1, pass_visual_frontend=False, verbose=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    rows = []
    paths = sorted((source / 'crops/scale1.7_none').glob('*/mouth.npz'))
    if len(paths) != 4:
        raise ValueError('Expected the four fixed original development clips')
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            frames, _ = prepare_frames(data['frames'], float(data['fps']), 96, None)
        # Cross the production chunk boundary (48), without excessive CPU memory.
        frames = frames[:53]
        if len(frames) != 53:
            raise ValueError('Audit needs 53 frames per clip')
        fixture = output / (path.parent.name + '.avi')
        writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*'FFV1'), 25, (96, 96))
        if not writer.isOpened():
            raise RuntimeError('Lossless FFV1 fixture writer unavailable')
        for frame in frames:
            writer.write(frame)
        writer.release()
        cap = cv2.VideoCapture(str(fixture))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        cap.release()
        decoded = np.stack(decoded)
        official = namespace['_parse_video'](str(fixture), True).permute(3, 0, 1, 2).unsqueeze(0)
        ours = normalize_available(decoded, np.ones(len(decoded), dtype=np.bool_))
        lengths = torch.tensor([len(decoded)])
        with torch.inference_mode():
            full_front = model.encoder.front(ours)
            chunked = chunk_frontend(model.encoder.front, ours)
            full_logits = model.decoder.ctcdecoder(model.medium(model.encoder(official, lengths)))
            chunk_logits = model.decoder.ctcdecoder(model.medium(
                model.encoder.back(model.encoder.projection(chunked), lengths)))
        row = dict(id=path.parent.name, input_sha256=sha256(path), frames=len(decoded),
            lossless_bgr_equal=bool(np.array_equal(frames, decoded)),
            normalization_max_abs=float((official-ours).abs().max()),
            normalization_close=bool(torch.allclose(official, ours, atol=1e-6, rtol=1e-5)),
            frontend_max_abs=float((full_front-chunked).abs().max()),
            frontend_close=bool(torch.allclose(full_front, chunked, atol=1e-5, rtol=1e-5)),
            logits_max_abs=float((full_logits-chunk_logits).abs().max()),
            logits_close=bool(torch.allclose(full_logits, chunk_logits, atol=1e-4, rtol=1e-4)),
            greedy_equal=collapse_ctc(full_logits[0].argmax(-1).tolist()) == collapse_ctc(chunk_logits[0].argmax(-1).tolist()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    hangul_failures = sum(grp2char(''.join(char2grp(chr(code)))) != chr(code) for code in range(44032, 55204))
    phrases = ['한글 입모양 자막.', '각나 간다 값이 높다!', '가나다 라마바', '힣 끝?']
    phrase_failures = sum(grp2char(''.join(char2grp(text))) != text for text in phrases)
    report = dict(schema_version=1, checkpoint_sha256=sha256(checkpoint),
        upstream_sources={str(p.relative_to(repo)): sha256(p) for p in
            [dataset, repo/'vocabulary/utils.py', repo/'avsr/models/encoder.py', repo/'preprocess.py']},
        torch_version=torch.__version__, opencv_version=cv2.__version__, device='cpu', threads=2,
        state_keys=len(state), parameters=sum(p.numel() for p in model.parameters()),
        tolerances=dict(normalization=[1e-6, 1e-5], frontend=[1e-5, 1e-5], logits=[1e-4, 1e-4]),
        reference_used=False, audio_used=False, rows=rows,
        hangul_roundtrip=dict(tested=11172, failed=hangul_failures, phrases_tested=len(phrases), phrases_failed=phrase_failures),
        limitations=['53-frame prefixes of four existing development clips, not full-clip accuracy evaluation',
            'Lossless transport isolates color/layout/normalization; does not reproduce training MP4 compression or annotated lip boxes',
            'Valid Hangul roundtrip does not validate malformed predicted grapheme sequences'])
    report['passed'] = not hangul_failures and not phrase_failures and all(
        all(row[key] for key in ['lossless_bgr_equal', 'normalization_close', 'frontend_close', 'logits_close', 'greedy_equal']) for row in rows)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Contract audit failed; inspect report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.checkpoint, args.source, args.output)
