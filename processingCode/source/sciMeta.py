# coding: utf-8

# author: Brian R. Pauw

import h5py
from scicatbam import scicatBam, h5py_casting
import logging
from collections import abc,OrderedDict

def update_deep(dictionary, path_update):
    """
    Update the main metadata dictionary with the new dictionary.
    """
    k = list(path_update.keys())[0]
    v = list(path_update.values())[0]
    if k not in dictionary.keys():
        dictionary[k] = v
    else:
        key_next = list(path_update[k].keys())[0]
        if key_next in dictionary[k].keys():
            dictionary[k] = update_deep(dictionary.get(k, {}), v)
        else:
            dictionary[k].update(v)
    return dictionary

def build_dictionary( levels, update_data):
    """"
    Creates a json-like level based dictionary for the whole path starting from /entry1 or whatever the first child of the root in the datatree is.
    """
    for level in levels[::-1]:
        update_data = dict({level:update_data})
    return update_data


def unwind(h5f, parent_path, metadata, default = 'none', leaveAsArray = False):
    """
    Current_level is the operating level, that is one level higher that the collected data.
    
    """
    if isinstance(h5f.get(parent_path), abc.Mapping):
        new_keys = h5f.get(parent_path).keys()
        for nk in sorted(new_keys):
            unwind(h5f, '/'.join([parent_path, nk]), metadata)
    else:
        try:
            val = h5f.get(parent_path)[()]
            val = h5py_casting(val,leaveAsArray)
        except OSError:
            logging.warning("has no value at path, setting to default")
            val = default
        except TypeError:
            logging.warning("cannot get value from file path {}, setting to default".format(parent_path))
            val = default

        attributes = {'value':val}
        try:
            attributes_add = h5f.get(parent_path).attrs
            a_key = attributes_add.keys()
            a_value = []
            for v in attributes_add.values():
                v = h5py_casting(v,leaveAsArray)
                a_value.append(v)
            attributes.update(dict(zip(a_key, a_value)))
        except KeyError as e:
            logging.warning(e)
            
        levels = parent_path.split('/')[1:]
        if list(attributes.keys()) == ['value']:# no attributes here
            nested_dict = val
        else:
            nested_dict = attributes.copy()
        if val != '':
            update_dict = build_dictionary(levels,nested_dict)
            metadata = update_deep(metadata, update_dict)

            
def unwind_flat(h5f, parent_path, metadata, default = 'none', leaveAsArray = False):
    """
    Updates the metadata with the following level. Current_level is the operating level, that is one level higher that the collected data.
    
    """
    if isinstance(h5f.get(parent_path), abc.Mapping):
        new_keys = h5f.get(parent_path).keys()
        for nk in sorted(new_keys):
            unwind_flat(h5f, '/'.join([parent_path, nk]), metadata)
    else:
        try:
            val = h5f.get(parent_path)[()]
            val = h5py_casting(val,leaveAsArray)
        except OSError:
            logging.warning("has no value at path, setting to default")
            val = default
        except TypeError:
            logging.warning("cannot get value from file path {}, setting to default".format(parent_path))
            val = default

        attributes = {'value':val}
        try:
            attributes_add = h5f.get(parent_path).attrs
            a_key = attributes_add.keys()
            a_value = []
            for v in attributes_add.values():
                v = h5py_casting(v,leaveAsArray)
                a_value.append(v)
            attributes.update(dict(zip(a_key, a_value)))
        except KeyError as e:
            logging.warning(e)
            
        levels = parent_path.split('/')[1:]
        if list(attributes.keys()) == ['value']:# no attributes here
            nested_dict = dict({'value':val})
        else:
            nested_dict = attributes.copy()
        if val != '':
            nested_dict['absolute path'] = '/'.join(levels)
            key = levels[-1]
            if key in metadata.keys():
                path_key_new = '/'.join(levels)
                path_key_old = metadata[key]['absolute path']
                for i in range(len(path_key_new)):
                    if path_key_new[i]!=path_key_old[i]:
                        i_start = i
                        break
                for i in range(len(path_key_new)):
                    if path_key_new[::-1][i]!=path_key_old[::-1][i]:
                        i_end = i
                        break
                if len(path_key_new[i_start:-i_end])==1:
                    key_new = path_key_new[:i_start+1].split('/')[-1]
                    key_old = path_key_old[:i_start+1].split('/')[-1]
                else:    
                    key_new = path_key_new[i_start:-i_end].split('/')[0]
                    key_old = path_key_old[i_start:-i_end].split('/')[0]
                
                
                metadata['.'.join(filter(None,[key, key_old]))] = metadata[key]
                metadata['.'.join(filter(None,[key, key_new]))]  = nested_dict
                if key_old !='':
                    del metadata[key] 
                
            else:
                metadata[key] = nested_dict

def create_meta(filename):
    """
    Opens nexus file and unwinds the directories to add up all the metadata and respective attributes
    """
    with h5py.File(filename, "r") as h5f:
        prior_keys = list(h5f.keys())
        if 'Saxslab' in prior_keys:
            prior_keys.remove('Saxslab')
        metadata = dict()#.fromkeys(prior_keys)
        parent_path = ''
        for pk in sorted(prior_keys):
            unwind(h5f, '/'.join([parent_path, pk]),metadata)
        if len(metadata.keys())==1:
            return metadata[list(metadata.keys())[0]]
        else:
            return metadata

def create_short_meta(filename, key_list, default = 'none', leaveAsArray = False):
    json_like = dict()
    with h5py.File(filename, "r") as h5f:
        for key in key_list:
            try:
                val = h5f[key][()] 
                val = h5py_casting(val, False)
            except KeyError as e:
                logging.warning("cannot get value from file path {}, setting to default".format(key))
                val = default
            try:
                units = h5f[key].attrs['units']
                if isinstance(units, bytes):
                    units = units.decode('UTF-8')
            except KeyError as e:
                logging.warning(e)
                units = '-'
            if key.split('/')[-1] in json_like.keys():
                json_like[' '.join([key.split('/')[-1], json_like[key.split('/')[-1]]['checker']])] = json_like[key.split('/')[-1]]
                json_like[' '.join([key.split('/')[-1],  key.split('/')[2]])] = { 'Value':val, 'Units':units, 'checker': key.split('/')[2] }
                del json_like[key.split('/')[-1]] 
                
            else:
                json_like[key.split('/')[-1]] = { 'Value':val, 'Units':units, 'checker': key.split('/')[2] }
        for key in json_like.keys():
            json_like[key] = {'Value':json_like[key]['Value'], 'Units':json_like[key]['Units']}
        return json_like

def create_meta_flat(filename):
    """
    Opens nexus file and unwinds the directories to add up all the metadata and respective attributes
    """
    with h5py.File(filename, "r") as h5f:
        prior_keys = list(h5f.keys())
        prior_keys.remove('Saxslab')
        
        parent_path = ''
        whole_meta = dict()
        for pk in sorted(prior_keys):
            metadata = dict()
            unwind_flat(h5f, '/'.join([parent_path, pk]),metadata)
            whole_meta[pk] = OrderedDict(sorted(metadata.items()))
        if len(whole_meta.keys())==1:
            whole_meta = whole_meta[list(whole_meta.keys())[0]]
            return whole_meta
        else:
            return whole_meta