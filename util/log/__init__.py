"""
log
===

Logging package. Used for runtime diagnostics.

Usage
-----
This package is intended to be used to send messages to the console after completing each significant task to enable
the user to see what the program is doing at the moment. To log a debug message to the console::

    >>> log.d("context", "data")
    12:30:50 | file_name, caller, line number | context: data.

To log an error message to the console::

    >>> log.e("context", "data")
    12:30:50 | file_name, caller, line number | context: data.

The separation of message types allows the user to toggle different kinds of outputs. Use the  flags to toggle
the logger output::

    >>> log.DEBUG = True
    >>> log.d("context", "data")
    21:54:52 | <input>, __main__, line 1 | context: data.
    >>> log.DEBUG = False
    >>> log.d("context", "data")
    ...
"""

from .log import e
from .log import d
