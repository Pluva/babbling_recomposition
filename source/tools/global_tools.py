"""
	File containing tools used throughout the system.
	_inverse_mapping_uktl:
	_inverse_mapping_ltuk:
	inverse_mapping:
	add_prefix_to_keys:
	get_major_in_list:
	get_best_in_list:
"""

#python
from collections import Counter

#local



# ------------------ </LISTS> ------------------ #


def _inverse_mapping_uktl(mapping):
    """
        Inverse the mapping of a dictionnary.
        Go from unique ID -> label to label -> [id1, id2, ...]

        :param mapping: The dictionnary to inverse.
		:type mapping: dict
		:return: The inversed dictionnary.
		:rtype: dict 
    """
    inv_mapping = {}
    for k, v in mapping.iteritems():
        inv_mapping.setdefault(v, []).append(k)
    return inv_mapping

def _inverse_mapping_ltuk(mapping):
    """
        Opposite function to __inverse_classes.
        Go from unique label -> [id1, id2, ...] to unique ID -> label

        :param mapping: The dictionnary to inverse.
		:type mapping: dict
		:return: The inversed dictionnary.
		:rtype: dict 
    """
    rev_mapping = {}
    for k, v in mapping.iteritems():
        for vi in v:
            rev_mapping[vi] = k 
    return rev_mapping

_INV_MAP_UKTL = 'unique_key_to_label'
_INV_MAP_LTUK = 'label_to_unique_key'

def inverse_mapping(mapping, mode):
	"""
		Inverse the mapping of a dictionnary.
		Mode will chose how to handle the mapping, from the two currently supported behavior.

		:param mapping: The dictionnary to inverse.
		:param mode: Way of inversing, depending on the given mapping.
		:type mapping: dict
		:type mode: string ['unique_key_to_label', 'label_to_unique_key']
		:return: The inversed dictionnary.
		:rtype: dict 
	"""
	if mode == 'unique_key_to_label':
		return _inverse_mapping_uktl(mapping)
	elif mode == 'label_to_unique_key':
		return _inverse_mapping_ltuk(mapping)


def add_prefix_to_keys(mapping, prefix):
    """
        Add the given prefix to all keys of this mapping.
        Keys are supposed to be string.

        :param mapping: Dictionnary to update.
        :param prefix: Prefix to add.
        :type mapping: python dict
        :type prefix: string
        :return: An updated copy of the mapping.
        :rtype: python dict
    """
    ret = {}
    for k, v in mapping.items():
        ret[prefix+k] = v
    return ret


def get_major_in_list(collec):
	"""
		Count the number of iteration of each element of this list, and returns the most present one.
		In case of equality, returns the first occuring.

		:param collec: The list to enumerate.
		:type collec: list
		:return: The value of the most present item.
		:rtype: depends of the inputed list.
	"""

        counts = Counter(collec)
        maximum = 0
        for k, v in counts.iteritems():
            if v > maximum:
                ret = k
                maximum = v
        return ret


def get_best_in_list(collec, ordered_values):
	"""
		Return the best value found in the given collection as ordered in the ordered_values parameter.
		An error if none values of ordered_values are in collec.

		:param collec: Collection in which to find the best value.
		:param ordered_values: Values ordered from best to worst.
		:type collec: list[type v]
		:type ordered_values: list[type v]
		:return: The first encoutered value from ordered_values.
		:rtype: type v
	"""

	best = None
	for v in ordered_values:
		if v in collec:
			best = v
			break

	if best is None:
		print('Given collection is: {}'.format(collec))
		print('Given ordered values are: {}'.format(ordered_values))
		raise ValueError('The given collection does not contains any values from ordered_values')

	return best


# ------------------ <LISTS/> ------------------ #