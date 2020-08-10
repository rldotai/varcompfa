"""Utilities for working with data, e.g., CSV files, JSON files, and so on."""
import io
import os
# import pickle
import pathlib
import zlib
import dill as pickle
import json_tricks as jt
import msgpack
import pandas as pd
import varcompfa


def save_agent(agent, path, overwrite=False):
    """Save an agent."""
    # Handle path being a path or file-like object
    # If it's a path, check if it's got an extension already else append `pkl`
    if not isinstance(agent, varcompfa.Agent):
        raise Exception("This function is for saving agents...")

    # Accomodate not specifying an extension
    base, ext = os.path.splitext(path)
    if ext == '':
        path = base + '.pkl'

    if os.path.exists(path) and not overwrite:
        raise Exception("File exists at: %s" % (path,))
    dump_pickle(agent, path)

def load_agent(path):
    """Load an agent."""
    # Handle path being a path or file-like object
    # If it's a path, check if it's got an extension already,
    # otherise try after appending `pkl`
    base, ext = os.path.splitext(path)
    if ext == '':
        path = base + '.pkl'
    agent = load_pickle(path)
    if not isinstance(agent, varcompfa.Agent):
        raise Exception("This function is for loading agents...")
    return agent

def dump_pickle(obj, path_or_buf=None):
    elem = pickle.dumps(obj, protocol=4)
    data = zlib.compress(elem)
    if path_or_buf is None:
        return data
    elif isinstance(path_or_buf, str):
        # Accomodate not specifying an extension
        base, ext = os.path.splitext(path_or_buf)
        if ext == '':
            path_or_buf = base + '.pkl'
        with open(path_or_buf, 'wb') as fh:
            fh.write(data)
    else:
        path_or_buf.write(data)

def load_pickle(path_or_buf):
    """Load a pickled object."""
    def loader(path_or_buf):
        """Load either from a buffer or a file path."""
        if isinstance(path_or_buf, str):
            base, ext = os.path.splitext(path_or_buf)
            if ext == '':
                alt_path = base + '.pkl'
            else:
                alt_path = None
            try:
                # Accomodate not specifying an extension
                if os.path.exists(path_or_buf):
                    exists = True
                elif alt_path and os.path.exists(alt_path):
                    exists = True
                    path_or_buf = alt_path
                else:
                    exists = False
            except (TypeError, ValueError):
                exists = False

            # If it's a filepath, open and read it, else treat as bytes
            if exists:
                return open(path_or_buf, 'rb').read()
            else:
                return bytes(path_or_buf, 'ascii')
        # If it's a pathlib Path, try loading it
        if isinstance(path_or_buf, pathlib.Path):
            return open(path_or_buf, 'rb').read()
        # If it's a bytes object, just return it
        if isinstance(path_or_buf, bytes):
            return path_or_buf

        # Buffer-like
        if hasattr(path_or_buf, 'read') and callable(path_or_buf.read):
            return path_or_buf.read()
        raise ValueError("Could not load `path_or_buf`")
    elem = zlib.decompress(loader(path_or_buf))
    return pickle.loads(elem)

def dump_msgpack(obj, path_or_buf=None, **kwargs):
    """Dump an object to msgpack, using zlib for compression."""
    # Dump the object using its built-in method or msgpack's default
    if hasattr(obj, 'to_msgpack') and callable(obj.to_msgpack):
        msg = obj.to_msgpack(**kwargs)
    else:
        msg = msgpack.dumps(obj)
    elem = zlib.compress(msg)

    if path_or_buf is None:
        return elem
    elif isinstance(path_or_buf, str):
        with open(path_or_buf, 'wb') as fh:
            fh.write(elem)
    else:
        path_or_buf.write(elem)

def load_msgpack(path_or_buf, use_pandas=True):
    """Load a compressed msgpack object, by default using Pandas"""
    def loader(path_or_buf):
        """Load either from a buffer or a file path."""
        if isinstance(path_or_buf, str):
            try:
                exists = os.path.exists(path_or_buf)
            except (TypeError, ValueError):
                exists = False

            # If it's a filepath, open and read it, else treat as bytes
            if exists:
                return open(path_or_buf, 'rb').read()
            else:
                return bytes(path_or_buf, 'ascii')

        # If it's a bytes object, just return it
        if isinstance(path_or_buf, bytes):
            return path_or_buf

        # Buffer-like
        if hasattr(path_or_buf, 'read') and callable(path_or_buf.read):
            return path_or_buf.read()
        raise ValueError("Could not load `path_or_buf`")

    elem = zlib.decompress(loader(path_or_buf))
    if use_pandas:
        return pd.read_msgpack(elem)
    else:
        return msgpack.loads(elem)

def dump_json(obj, path_or_buf=None, **kwargs):
    """Serialize and compress an object using JSON."""
    if hasattr(obj, 'to_json') and callable(obj.to_json):
        msg = obj.to_json(**kwargs)
    else:
        msg = jt.dumps(obj)
    elem = zlib.compress(bytes(msg, 'ascii'))

    if path_or_buf is None:
        return elem
    elif isinstance(path_or_buf, str):
        with open(path_or_buf, 'wb') as fh:
            fh.write(elem)
    else:
        path_or_buf.write(elem)

def load_json(path_or_buf, use_pandas=True):
    """Load a compressed JSON object, by default using Pandas"""
    def loader(path_or_buf):
        """Load either from a buffer or a file path."""
        if isinstance(path_or_buf, str):
            try:
                exists = os.path.exists(path_or_buf)
            except (TypeError, ValueError):
                exists = False

            # If it's a filepath, open and read it, else treat as bytes
            if exists:
                return open(path_or_buf, 'rb').read()
            else:
                return bytes(path_or_buf, 'ascii')

        # If it's a bytes object, just return it
        if isinstance(path_or_buf, bytes):
            return path_or_buf

        # Buffer-like
        if hasattr(path_or_buf, 'read') and callable(path_or_buf.read):
            return path_or_buf.read()
        raise ValueError("Could not load `path_or_buf`")

    elem = zlib.decompress(loader(path_or_buf))
    if use_pandas:
        return pd.read_json(elem)
    else:
        return jt.loads(str(elem, 'ascii'))


def to_csv(obj, output=None, metadata=dict()):
    """Convert an object to a CSV file.

    TODO: Follow `pandas` insofar as possible except allowing for serializing
    numpy arrays and preserving metadata in comments in the first few lines.
    """
    pass


def seq_to_dict(inner_seq, keys):
    """Convert a sequence of sequences to a sequence of dicts with each dict
    associating each element of the inner sequence to a key."""
    return [{k: elem for k, elem in zip(keys, inner_seq)} for inner_seq in seq]

def string_to_array(s, sep=' ', shape=None, **kwargs):
    """Convert a string to a numpy array.
    For example, `"[1 2 3]"` --> `np.array([1, 2, 3])`

    Useful for working with pandas and CSV arrays.
    """
    # Replace brackets because numpy doesn't like them
    raw = s.replace('[', '').replace(']', '')
    arr = np.fromstring(raw, sep=sep)
    # Optionally reshape
    if shape is not None:
        return arr.reshape(shape)
    else:
        return arr

def make_hashable(df):
    """Make a DataFrame's columns hashable in order to support groupby or other
    indexing operations.
    """
    # dtypes
    # If it's a column of dicts, try to expand into new columns
    # and drop the old ones


def load_df(contexts):
    """Load contexts into a DataFrame with some preprocessing"""
    ret = pd.DataFrame(contexts)
    from numbers import Number

    def make_hashable(elem):
        """Try to make an item hashable."""
        if isinstance(elem, Number):
            return elem
        elif isinstance(elem, np.ndarray) and elem.squeeze().ndim == 0:
            return elem.item()
        else:
            return tuple(elem)

    # Make it possible to hash (and therefore group) certain columns
    ret['obs'] = ret['obs'].apply(make_hashable)
    ret['obs_p'] = ret['obs_p'].apply(make_hashable)

    return ret
