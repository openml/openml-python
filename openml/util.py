import sys

if sys.version_info[0] < 3:
    from urllib2 import URLError
else:
    from urllib.error import URLError

import six

oml_cusual_string = r'([a-zA-Z0-9_\-,\.\(\)])+'

def is_string(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

__all__ = ['URLError', 'is_string', 'oml_cusual_string']
