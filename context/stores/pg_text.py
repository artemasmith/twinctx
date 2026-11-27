"""Reversible text storage for PostgreSQL, which cannot store NUL in text."""


def project_text(value):
    if not isinstance(value, str):
        raise ValueError('atom text must be a string')
    return (value.replace('\x00', ' '), value.encode('utf-8')) if '\x00' in value else (value, None)


def restore_text(projected, raw):
    if raw is None:
        return projected
    value = bytes(raw).decode('utf-8')
    if '\x00' not in value or value.replace('\x00', ' ') != projected:
        raise ValueError('stored raw atom text does not match its search projection')
    return value
