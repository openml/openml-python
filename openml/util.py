import sys

if sys.version_info[0] < 3:
    from urllib2 import URLError
else:
    from urllib.error import URLError


def is_string(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

def version_complies(major, minor=None):
    version = sys.version_info
    if version[0] > major:
        return True
    if version[0] < major:
        return False
    # version == major
    if minor is None or version[1] >= minor:
        return True
    return False

__all__ = ['URLError', 'is_string']
