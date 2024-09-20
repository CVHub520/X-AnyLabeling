import datetime
import logging
import termcolor

COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        # Ensure the message is formatted with the provided arguments
        record.message = record.getMessage()

        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored(f"{record.levelname:<7}")
            record.message2 = colored(record.message)

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")

        return super().format(record)


class ColoredLogger(logging.Logger):
    FORMAT = (
        "[%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
    )

    def __init__(self, name):
        super().__init__(name, logging.INFO)

        color_formatter = ColoredFormatter(self.FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)


# Ensure logger is an instance of ColoredLogger
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("X-AnyLabeling")
