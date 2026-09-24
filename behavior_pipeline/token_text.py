from __future__ import annotations

"""Tokenizer-independent reconstruction of compositional token text pieces."""

import codecs


def _byte_level_decoder() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(byte_values)
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + offset)
            offset += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


_BYTE_LEVEL_DECODE = _byte_level_decoder()
_BYTE_LEVEL_BOUNDARY_MARKERS = frozenset({"Ġ", "Ċ", "ĉ"})
_BYTE_LEVEL_TOKENIZER_CACHE: dict[int, bool] = {}
_BYTE_LEVEL_TOKENIZER_OWNERS: dict[int, object] = {}


def _uses_byte_level_vocabulary(tokenizer) -> bool:
    key = id(tokenizer)
    _BYTE_LEVEL_TOKENIZER_OWNERS[key] = tokenizer
    cached = _BYTE_LEVEL_TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        vocabulary = tokenizer.get_vocab()
        result = any(
            marker in str(token)
            for token in vocabulary
            for marker in _BYTE_LEVEL_BOUNDARY_MARKERS
        )
    except Exception:
        result = False
    _BYTE_LEVEL_TOKENIZER_CACHE[key] = result
    return result


def _byte_level_piece_bytes(token: str) -> bytes:
    encoded = bytearray()
    for character in str(token):
        value = _BYTE_LEVEL_DECODE.get(character)
        if value is None:
            encoded.extend(character.encode("utf-8"))
        else:
            encoded.append(value)
    return bytes(encoded)


def token_id_texts(tokenizer, token_ids) -> list[str]:
    """Decode token IDs into compositional display pieces.

    Some DeepSeek/Qwen tokenizer combinations return non-compositional text for
    decode of one token: whitespace markers disappear and characters can be
    assigned to the wrong token. Byte-level vocabulary pieces are compositional,
    so decode those pieces directly whenever that vocabulary is detected.
    """
    ids = [int(value) for value in token_ids]
    if not ids:
        return []
    try:
        raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        raw_tokens = None
    if (
        isinstance(raw_tokens, list)
        and len(raw_tokens) == len(ids)
        and _uses_byte_level_vocabulary(tokenizer)
    ):
        special_ids = {
            int(value) for value in (getattr(tokenizer, "all_special_ids", []) or [])
        }
        values: list[str] = []
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for token_id, token in zip(ids, raw_tokens):
            if token_id in special_ids:
                pending = decoder.decode(b"", final=True)
                if pending and values:
                    values[-1] += pending
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                try:
                    values.append(
                        str(
                            tokenizer.decode(
                                [token_id],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    )
                    continue
                except Exception:
                    pass
            values.append(decoder.decode(_byte_level_piece_bytes(str(token)), final=False))
        pending = decoder.decode(b"", final=True)
        if pending and values:
            values[-1] += pending
        return values
    values = []
    for token_id in ids:
        try:
            value = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            value = str(token_id)
        values.append(str(value))
    return values


def decode_token_ids(tokenizer, token_ids) -> str:
    """Decode IDs consistently with :func:`token_id_texts`."""
    ids = [int(value) for value in token_ids]
    pieces = token_id_texts(tokenizer, ids)
    try:
        raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        raw_tokens = None
    if isinstance(raw_tokens, list) and _uses_byte_level_vocabulary(tokenizer):
        return "".join(pieces)
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return "".join(pieces)
