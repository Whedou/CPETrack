import os
import re

import torch


_WORD_VOCAB = [
    "umbrella", "backpack", "suitcase", "cameraman", "motorcycle", "bicycle",
    "person", "woman", "women", "man", "men", "girl", "boy", "child", "baby",
    "car", "truck", "bus", "bike", "biker", "rider", "runner", "player",
    "dog", "cat", "horse", "ball", "book", "door", "mirror", "face", "head",
    "hand", "hat", "bag", "box", "cup", "bottle", "phone", "chair", "table",
    "yellow", "black", "white", "red", "green", "blue", "gray", "grey",
    "orange", "brown", "pink", "purple", "dark", "light",
    "left", "right", "front", "back", "side", "around", "near", "on", "in",
    "with", "at", "under", "over", "open", "close", "grass", "road", "street",
    "running", "walking", "riding", "standing", "sitting", "moving",
]

_WORD_VOCAB = sorted(set(_WORD_VOCAB), key=len, reverse=True)


def _split_known_words(token):
    """Greedy split for compact LasHeR-style sequence names."""
    words = []
    i = 0
    while i < len(token):
        match = None
        for word in _WORD_VOCAB:
            if token.startswith(word, i):
                match = word
                break
        if match is None:
            j = i + 1
            while j < len(token) and not any(token.startswith(word, j) for word in _WORD_VOCAB):
                j += 1
            words.append(token[i:j])
            i = j
        else:
            words.append(match)
            i += len(match)
    return words


def clean_sequence_name(name):
    """
    Convert a dataset sequence/folder name into a weak semantic text key.

    Examples:
        yellowumbrellagirl -> yellow umbrella girl
        cameraman_1202 -> cameraman
        womanaroundcar -> woman around car
    """
    if name is None:
        return ""

    base = os.path.basename(str(name).replace("\\", "/"))
    base = os.path.splitext(base)[0].lower()
    base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\d+", " ", base)
    base = re.sub(r"[^a-z\s]", " ", base)

    words = []
    for token in base.split():
        if len(token) <= 2:
            words.append(token)
        else:
            words.extend(_split_known_words(token))

    return " ".join(w for w in words if w).strip()


class TextFeatureCache:
    """
    Optional lookup table from sequence/text names to precomputed text features.

    Expected cache formats:
        {"yellowcar": tensor(...), "yellow car": tensor(...)}
        {"features": {"yellowcar": tensor(...)}, "feature_dim": 512}
    """

    def __init__(self, enabled=False, feature_path="", feature_dim=512, dropout=0.0):
        self.enabled = bool(enabled)
        self.feature_path = feature_path or ""
        self.feature_dim = int(feature_dim)
        self.dropout = float(dropout)
        self.features = {}

        if self.enabled and self.feature_path:
            self._load(self.feature_path)

    @classmethod
    def from_cfg(cls, cfg):
        if cfg is None:
            return cls(False)
        return cls(
            enabled=getattr(cfg, "ENABLE", False),
            feature_path=getattr(cfg, "FEATURE_PATH", ""),
            feature_dim=getattr(cfg, "FEATURE_DIM", 512),
            dropout=getattr(cfg, "DROPOUT", 0.0),
        )

    def _load(self, feature_path):
        path = os.path.expanduser(feature_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("Text feature cache not found: {}".format(path))

        cache = torch.load(path, map_location="cpu")
        if isinstance(cache, dict) and "features" in cache:
            if "feature_dim" in cache:
                cache_dim = int(cache["feature_dim"])
                if cache_dim != self.feature_dim:
                    raise ValueError(
                        "Text cache feature_dim {} does not match config FEATURE_DIM {}".format(
                            cache_dim, self.feature_dim
                        )
                    )
            cache = cache["features"]

        if not isinstance(cache, dict):
            raise TypeError("Text feature cache must be a dict, got {}".format(type(cache)))

        self.features = {}
        for key, value in cache.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
            self.features[str(key).lower()] = tensor
            clean_key = clean_sequence_name(key)
            if clean_key:
                self.features.setdefault(clean_key, tensor)

        if self.features:
            loaded_dim = int(next(iter(self.features.values())).numel())
            if loaded_dim != self.feature_dim:
                raise ValueError(
                    "Text feature vector dim {} does not match config FEATURE_DIM {}".format(
                        loaded_dim, self.feature_dim
                    )
                )

    def _feature_for_name(self, name):
        if not self.enabled:
            return None

        candidates = []
        if name is not None:
            raw = str(name).lower()
            candidates.extend([raw, clean_sequence_name(raw)])

        for key in candidates:
            if key in self.features:
                return self.features[key]

        return torch.zeros(self.feature_dim, dtype=torch.float32)

    def get_batch(self, names, device=None, training=False):
        if not self.enabled:
            return None

        if isinstance(names, (str, bytes)) or names is None:
            names = [names]

        feats = [self._feature_for_name(name) for name in names]
        batch = torch.stack(feats, dim=0)

        if training and self.dropout > 0:
            keep = torch.rand(batch.shape[0], 1) >= self.dropout
            batch = batch * keep.to(dtype=batch.dtype)

        if device is not None:
            batch = batch.to(device)
        return batch
