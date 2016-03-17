from collections import OrderedDict
import re
import os
from ..exceptions import OpenMLCacheException
from .split import OpenMLSplit


def _get_cached_splits(api_connector):
    splits = OrderedDict()
    for task_cache_dir in [api_connector.task_cache_dir,
                           api_connector._private_directory_tasks]:
        directory_content = os.listdir(task_cache_dir)
        directory_content.sort()

        for filename in directory_content:
            match = re.match(r"(tid)_([0-9]*)\.arff", filename)
            if match:
                tid = match.group(2)
                tid = int(tid)

                splits[tid] = api_connector.get_cached_task(tid)

    return splits


def _get_cached_split(api_connector, tid):
    for task_cache_dir in [api_connector.task_cache_dir,
                           api_connector._private_directory_tasks]:
        try:
            split_file = os.path.join(task_cache_dir,
                                      "tid_%d.arff" % int(tid))
            split = OpenMLSplit.from_arff_file(split_file)
            return split

        except (OSError, IOError):
            continue

    raise OpenMLCacheException("Split file for tid %d not "
                               "cached" % tid)
