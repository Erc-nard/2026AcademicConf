"""Deterministic synthetic input-loss probes, applied after resize and before normalization."""

CONDITIONS = ('none', 'spatial25', 'spatial50', 'temporal25')


def corrupt_frames(frames, condition):
    import numpy as np
    if condition not in CONDITIONS:
        raise ValueError("Unknown synthetic corruption")
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3 or not len(frames):
        raise ValueError("Expected nonempty uint8 THWC frames")
    output = frames.copy()
    count, height, width, _ = frames.shape
    info = dict(condition=condition, fill_value=127, synthetic=True, masked_fraction=0.)
    if condition.startswith('spatial'):
        band = max(1, round(height * (0.25 if condition == 'spatial25' else .5)))
        start = (height-band)//2
        output[:, start:start+band] = 127
        info.update(masked_fraction=band/height, y_start=start, y_end=start+band,
                    frames_masked=count, pixels_per_masked_frame=band*width)
    elif condition == 'temporal25':
        band = max(1, round(count*.25))
        start = (count-band)//2
        output[start:start+band] = 127
        info.update(masked_fraction=band/count, frame_start=start, frame_end=start+band,
                    frames_masked=band, pixels_per_masked_frame=height*width)
    return output, info
