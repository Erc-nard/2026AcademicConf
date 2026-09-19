"""Input availability is not an annotation of actual face visibility or confidence."""


def resample_availability(mask, source_count, source_fps, output_count, output_fps):
    import numpy as np
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (source_count,):
        raise ValueError("ROI availability must be a boolean vector matching source frames")
    if min(source_count, output_count) < 1 or not np.isfinite([source_fps, output_fps]).all() or min(source_fps, output_fps) <= 0:
        raise ValueError("Invalid frame counts or FPS")
    indices = np.clip(np.floor((np.arange(output_count)+.5)*source_fps/output_fps).astype(int), 0, source_count-1)
    return mask[indices]


def normalize_available(frames, observed):
    """Keep timeline, normalize real ROIs only, set unavailable normalized pixels to zero."""
    import torch
    values = torch.as_tensor(frames).float()
    mask = torch.as_tensor(observed, dtype=torch.bool)
    if mask.ndim != 1 or len(mask) != len(values) or not mask.any():
        raise ValueError("At least one usable ROI and a matching mask are required")
    usable = values if mask.all() else values[mask]
    std = usable.std()
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Usable ROI has zero or invalid variance")
    output = (values-usable.mean())/std
    output[~mask] = 0
    return output.permute(3, 0, 1, 2).unsqueeze(0)
