import xmltodict
import six
from ._api_calls import _perform_api_call


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


def _tag_entity(entity_type, entity_id, tag, untag=False):
    """Abstract function that can be used as a partial for tagging entities
       on OpenML

        Parameters
        ----------
        entity_type : str
            Name of the entity to tag (e.g., run, flow, data)

        entity_id : int
            OpenML id of the entity

        tag : str
            The tag

        untag : bool
            Set to true if needed to untag, rather than tag

        Returns
        -------
        tags : list
            List of tags that the entity is still tagged with
        """
    legal_entities = {'data', 'task', 'flow', 'setup', 'run'}
    if entity_type not in legal_entities:
        raise ValueError('Can\'t tag a %s' %entity_type)

    uri = '%s/tag' %entity_type
    main_tag = 'oml:%s_tag' %entity_type
    if untag:
        uri = '%s/untag' %entity_type
        main_tag = 'oml:%s_untag' %entity_type


    post_variables = {'%s_id'%entity_type: entity_id, 'tag': tag}
    result_xml = _perform_api_call(uri, post_variables)

    result = xmltodict.parse(result_xml, force_list={'oml:tag'})[main_tag]

    if 'oml:tag' in result:
        return result['oml:tag']
    else:
        # no tags, return empty list
        return []