import inspect
from datetime import datetime
from colorama import Fore

DEBUG = False
ERROR = True


def d(context, data):
    if DEBUG:
        print('{} {} {} {} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                        Fore.LIGHTBLACK_EX + " |",
                                        Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[-1] + ",",
                                        Fore.LIGHTYELLOW_EX + inspect.getmodule(inspect.stack()[1][0]).__name__ + ',' +
                                        Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                        Fore.LIGHTBLACK_EX + " |",
                                        Fore.LIGHTBLUE_EX + context + ":",
                                        Fore.BLUE + data + "."))


def e(context, data):
    if ERROR:
        print('{:} {:} {:} {:} {:<}'.format(Fore.LIGHTGREEN_EX + datetime.now().strftime("%H:%M:%S") +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTWHITE_EX + inspect.getfile(inspect.stack()[1][0]).rsplit("\\")[-1] + ",",
                                            Fore.LIGHTYELLOW_EX + inspect.getmodule(inspect.stack()[1][0]).__name__ + ',' +
                                            Fore.YELLOW + ' line ' + str(inspect.stack()[1].lineno) +
                                            Fore.LIGHTBLACK_EX + " |",
                                            Fore.LIGHTRED_EX + context + ":",
                                            Fore.RED + data + "."))
