"""
Logging module, used for generating formatted runtime messages to the console.
    Functions:
        d(context, data)
        e(context, data)

    Global variables:
        DEBUG : Flag to enable debug logs
        ERROR : Flag to enable error logs
"""

import inspect
from datetime import datetime
from colorama import Fore

DEBUG = False
"""Flag to enable debug logs"""
ERROR = True
"""Flag to enable error logs"""


def d(context: str, data: str) -> None:
    """
    Log a debug message to the logger. Prints several parameters of the state of the program, and any extra
    information provided by the user.
    :param context : str
    Brief context behind the debug message
    :param data : str
    Any other context or information worth printing
    :return: None
    """
    if DEBUG:
        print('{} {} {} {} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                        Fore.LIGHTBLACK_EX + " |",
                                        Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                            -1] + ",",
                                        Fore.LIGHTYELLOW_EX + inspect.getmodule(inspect.stack()[1][0]).__name__ + ',' +
                                        Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                        Fore.LIGHTBLACK_EX + " |",
                                        Fore.LIGHTBLUE_EX + context + ":",
                                        Fore.BLUE + data + "." + Fore.LIGHTWHITE_EX))


def e(context: str, data: str) -> None:
    """
    Log an error message to the logger. Prints several parameters of the state of the program, and any extra
    information provided by the user.
    :param context: Brief context behind the error message
    :param data: Any other context or information worth printing
    :return: None
    """
    if ERROR:
        print('{:} {:} {:} {:} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[
                                                -1] + ",",
                                            Fore.LIGHTYELLOW_EX + inspect.getmodule(
                                                inspect.stack()[1][0]).__name__ + ',' +
                                            Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTRED_EX + context + ":",
                                            Fore.RED + data + "." + Fore.LIGHTWHITE_EX))
