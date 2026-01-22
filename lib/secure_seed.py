from hashlib import md5


def extend_seed(base_seed = 42, cut_off_bytes=None):
    _hash = md5(base_seed.to_bytes(32))
    _hex = _hash.hexdigest()
    if cut_off_bytes:
        _hex = _hex[:cut_off_bytes*2]
    _int = int(_hex, 16)
    return _int