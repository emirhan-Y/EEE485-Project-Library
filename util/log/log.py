"""
Logging class, used for generating formatted runtime messages to the console.

Functions
---------
* debug_message(context, data): Log a debug message to the logger.
* error_message(context, data): Log an error message to the logger.

Variables
----------------
* DEBUG : Flag to enable debug logs
* ERROR : Flag to enable error logs
"""

import inspect
from datetime import datetime
from colorama import Fore


class log:
    def __init__(self, debug_enable: bool, error_enable: bool):
        """
        Create the log special to that class. It is recommended to use only one logger present at a time.

        Parameters
        ------------
        debug_enable : bool
            Set True to enable debug messages
        error_enable : bool
            Set True to enable error messages
        """
        self.__DEBUG = debug_enable
        """Flag to enable debug logs"""
        self.__ERROR = error_enable
        """Flag to enable error logs"""

    def enable_debug_log(self):
        self.__DEBUG = True

    def disable_debug_log(self):
        self.__DEBUG = False

    def enable_error_log(self):
        self.__ERROR = True

    def disable_error_log(self):
        self.__ERROR = False

    def close(self):
        print('{} {} {} {}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                   Fore.LIGHTBLACK_EX + " |",
                                   Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                       -1] + ",",
                                   Fore.LIGHTYELLOW_EX + inspect.getmodule(
                                       inspect.stack()[1][0]).__name__ + ',' +
                                   Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                   Fore.LIGHTBLACK_EX + " |",
                                   Fore.LIGHTRED_EX + 'logger disabled' + "."
                                   + Fore.LIGHTWHITE_EX))
        del self

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
        if self.__DEBUG:
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

        Returns
        -------
        None
        """
        if self.__ERROR:
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
