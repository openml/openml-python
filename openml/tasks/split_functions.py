from collections import OrderedDict
import re
import os
from ..exceptions import OpenMLCacheException
from .split import OpenMLSplit
from .. import config
from .task_functions import get_cached_task


def _get_cached_splits():
    splits = OrderedDict()
    for cache_dir in [config.cachedir, config.privatedir]:
        task_cache_dir = os.path.join(cache_dir, "tasks")
        directory_content = os.listdir(task_cache_dir)
        directory_content.sort()

        for filename in directory_content:
            match = re.match(r"(tid)_([0-9]*)\.arff", filename)
            if match:
                tid = match.group(2)
                tid = int(tid)

                splits[tid] = get_cached_task(tid)

    return splits


def _get_cached_split(tid):
    for cache_dir in [config.cachedir, config.privatedir]:
        task_cache_dir = os.path.join(cache_dir, "tasks")
        split_file = os.path.join(task_cache_dir,
                                  "tid_%d.arff" % int(tid))
        try:
            split = OpenMLSplit.from_arff_file(split_file)
            return split

        except (OSError, IOError):
            continue

    raise OpenMLCacheException("Split file for tid %d not "
                               "cached" % tid)
