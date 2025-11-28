# InfoLogger
#
# Author: Christian Sonnabend
# Contact: christian.sonnabend@cern.ch

# from termcolor import colored
import traceback
from datetime import datetime

def colored(text, color=None, on_color=None, attrs=None):
    """
    Minimal replacement for termcolor.colored.

    Parameters
    ----------
    text : str
        Text to colorize.
    color : str or None
        One of: 'grey', 'red', 'green', 'yellow', 'blue',
                'magenta', 'cyan', 'white'.
    on_color : str or None
        One of: 'on_grey', 'on_red', 'on_green', 'on_yellow',
                'on_blue', 'on_magenta', 'on_cyan', 'on_white'.
    attrs : list[str] or None
        Any of: 'bold', 'dark', 'underline', 'blink',
                'reverse', 'concealed'.
    """
    color_codes = {
        "grey": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
    }

    on_color_codes = {
        "on_grey": 40,
        "on_red": 41,
        "on_green": 42,
        "on_yellow": 43,
        "on_blue": 44,
        "on_magenta": 45,
        "on_cyan": 46,
        "on_white": 47,
    }

    attr_codes = {
        "bold": 1,
        "dark": 2,
        "underline": 4,
        "blink": 5,
        "reverse": 7,
        "concealed": 8,
    }

    codes = []

    if color is not None and color in color_codes:
        codes.append(str(color_codes[color]))

    if on_color is not None and on_color in on_color_codes:
        codes.append(str(on_color_codes[on_color]))

    if attrs:
        for a in attrs:
            if a in attr_codes:
                codes.append(str(attr_codes[a]))

    if not codes:
        return text

    start = "\033[" + ";".join(codes) + "m"
    end = "\033[0m"
    return f"{start}{text}{end}"


class logger():

    def __init__(self, task_name=None):
        self.severities_color = {
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "FATAL": "magenta"
        }
        self.severities_number = {v: k for k, v in enumerate(self.severities_color.keys())}
        self.task_name = task_name

    def debug(self, message):
        self.log_message(self.severities_number["DEBUG"], message)
    def info(self, message):
        self.log_message(self.severities_number["INFO"], message)
    def warning(self, message):
        self.log_message(self.severities_number["WARNING"], message)
    def error(self, message):
        self.log_message(self.severities_number["ERROR"], message)
    def fatal(self, message):
        self.log_message(self.severities_number["FATAL"], message)

    def log_message(self, level, message):
        if level < 0:
            level = 0
        elif level >= len(self.severities_color):
            level = len(self.severities_color) - 1
        severity_str = list(self.severities_color.keys())[level]
        color = self.severities_color[severity_str]
        print(colored(f"[{self.task_name}, {datetime.now()}] ({severity_str})", color) + f" {message}")
        if level >= 4:
            traceback.print_stack()
            raise Exception(f"Fatal error: {message}")

    def __call__(self, severity, message):
        """
        Prepare to log a message with the specified severity.

        :param severity: The severity level of the log message. Can be int or string.
        :return: A function that logs a message with the given severity.
        """
        if isinstance(severity, str) and severity in self.severities_color:
            severity_level = list(self.severities_color.keys()).index(severity)
            return self.log_message(severity_level, message)
        elif isinstance(severity, int):
            return self.log_message(severity, message)
