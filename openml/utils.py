import six

from openml.exceptions import OpenMLServerException


def extract_xml_tags(xml_tag_name, node, allow_none=True):
    """Helper to extract xml tags from xmltodict.

    Parameters
    ----------
    xml_tag_name : str
        Name of the xml tag to extract from the node.

    node : object
        Node object returned by ``xmltodict`` from which ``xml_tag_name``
        should be extracted.

    allow_none : bool
        If ``False``, the tag needs to exist in the node. Will raise a
        ``ValueError`` if it does not.

    Returns
    -------
    object
    """
    if xml_tag_name in node and node[xml_tag_name] is not None:
        if isinstance(node[xml_tag_name], dict):
            rval = [node[xml_tag_name]]
        elif isinstance(node[xml_tag_name], six.string_types):
            rval = [node[xml_tag_name]]
        elif isinstance(node[xml_tag_name], list):
            rval = node[xml_tag_name]
        else:
            raise ValueError('Received not string and non list as tag item')

        return rval
    else:
        if allow_none:
            return None
        else:
            raise ValueError("Could not find tag '%s' in node '%s'" %
                             (xml_tag_name, str(node)))
            
def list_all(listing_call, batch_size=10000, *args, **filters):
    """Helper to handle paged listing requests.

    Example usage:

    ``evaluations = list_all(list_evaluations, "predictive_accuracy", task=mytask)``

    Note: I wanted to make this a generator, but this is not possible since all
    listing calls return dicts
    
    Parameters
    ----------
    listing_call : callable
        Call listing, e.g. list_evaluations.
    batch_size : int (default: 10000)
        Batch size for paging.
    *args : Variable length argument list
        Any required arguments for the listing call.
    **filters : Arbitrary keyword arguments
        Any filters that can be applied to the listing function.
        
    Returns
    -------
    dict
    """
    page = 0
    result = {}

    while True:
        try:
            new_batch = listing_call(
                *args,
                size=batch_size,
                offset=batch_size*page,
                **filters
            )
        except OpenMLServerException as e:
            if page == 0 and e.args[0] == 'No results':
                raise e
            else:
                break
        result.update(new_batch)
        page += 1

    return result
