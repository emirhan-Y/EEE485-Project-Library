"""
Console interface class used for outputting user specified messages to the console.

Variables
----------------
* _DEBUG : Flag to enable debug messages
* _WARNING : Flag to enable warning messages
* _ERROR : Flag to enable error messages

Methods
-------
* debug_message(context, data): Log a debug message to the logger.
* warning_message(context, data): Log a warning message to the logger.
* error_message(context, data): Log an error message to the logger.
"""

import inspect
from datetime import datetime
from colorama import Fore


class log:
    def __init__(self, debug_enable, warning_enable, error_enable):
        self._DEBUG = debug_enable
        """Flag to enable debug messages"""
        self._WARNING = warning_enable
        """Flag to enable warning messages"""
        self._ERROR = error_enable
        """Flag to enable error messages"""

    def enable_debug_log(self):
        """
        Enable debug messages
        """
        self._DEBUG = True

    def disable_debug_log(self):
        """
        Disable debug messages
        """
        self._DEBUG = False

    def enable_warning_log(self):
        """
        Enable warning messages
        """
        self._WARNING = True

    def disable_warning_log(self):
        """
        Disable warning messages
        """
        self._WARNING = False

    def enable_error_log(self):
        """
        Enable error messages
        """
        self._ERROR = True

    def disable_error_log(self):
        """
        Disable error messages
        """
        self._ERROR = False

    def debug_message(self, context: str, data: str) -> None:
        """
        Log a debug message to the logger. Prints several parameters of the state of the program, and any extra
        information provided by the user.

        Parameters
        ------------
        context : str
            Brief context behind the debug message
        data : str
            Any other context or information worth printing
        """
        if self._DEBUG:
            print('{} {} {} {} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                                -1] + ",",
                                            Fore.LIGHTYELLOW_EX + inspect.getmodule(
                                                inspect.stack()[1][0]).__name__ + ',' +
                                            Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTBLUE_EX + context + ":",
                                            Fore.BLUE + data + "." + Fore.LIGHTWHITE_EX))

    def warning_message(self, context: str, data: str) -> None:
        """
        Log a warning message to the logger. Prints several parameters of the state of the program, and any extra
        information provided by the user.

        Parameters
        ------------
        context : str
            Brief context behind the warning message
        data : str
            Any other context or information worth printing
        """
        if self._WARNING:
            print('{} {} {} {} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                                -1] + ",",
                                            Fore.LIGHTYELLOW_EX + inspect.getmodule(
                                                inspect.stack()[1][0]).__name__ + ',' +
                                            Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTRED_EX + context + ":",
                                            Fore.LIGHTYELLOW_EX + data + "." + Fore.LIGHTWHITE_EX))

    def error_message(self, context: str, data: str) -> None:
        """
        Log an error message to the logger. Prints several parameters of the state of the program, and any extra
        information provided by the user.

        Parameters
        ------------
        context : str
            Brief context behind the error message
        data : str
            Any other context or information worth printing
        """
        if self._ERROR:
            print('{:} {:} {:} {:} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                                Fore.LIGHTBLACK_EX + " |",
                                                Fore.LIGHTWHITE_EX +
                                                inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                                    -1] + ",",
                                                Fore.LIGHTYELLOW_EX + inspect.getmodule(
                                                    inspect.stack()[1][0]).__name__ + ',' +
                                                Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                                Fore.LIGHTBLACK_EX + " |",
                                                Fore.LIGHTRED_EX + context + ":",
                                                Fore.RED + data + "." + Fore.LIGHTWHITE_EX))
