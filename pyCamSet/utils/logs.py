"""
Logging configuration for pyCamSet.

pyCamSet is a library, so it must not reconfigure the root logger of whatever
application imports it. Every module logs to a child of the ``pyCamSet``
logger, and :func:`setup_logging` installs a coloured handler on that logger
alone. A script that just calls ``calibrate_cameras`` gets the colours without
having to ask for them; an application that has already configured logging
keeps its own configuration and still receives pyCamSet's records by
propagation.
"""
from __future__ import annotations

import logging

import coloredlogs

from pyCamSet.utils.report_format import SOLARIZED

#: The logger every pyCamSet module hangs off.
LOGGER_NAME = "pyCamSet"

# The coloredlogs default format spends about sixty columns on a date,
# hostname, logger name and pid before the message starts, which wraps the
# calibration results onto a second line in a normal width terminal.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# The palette is shared with the report blocks, so a warning in the log and an
# amber cell in a table are the same colour. See pyCamSet.utils.report_format
# for why it is Solarized and why it is 256 colour indices.
#
# Readability never rests on hue alone. INFO is most of the output and is left
# in the terminal's own foreground, so it cannot clash with anything, and the
# level name is bold so it reads even where a hue is washed out. Severity then
# escalates through three accents: yellow, orange, red.
LOG_FIELD_STYLES = {
    "asctime": {"color": SOLARIZED["base01"]},
    "levelname": {"bold": True},
    "name": {"color": SOLARIZED["base01"]},
}
LOG_LEVEL_STYLES = {
    "debug": {"color": SOLARIZED["base01"]},
    "info": {},
    "warning": {"color": SOLARIZED["yellow"]},
    "error": {"color": SOLARIZED["orange"]},
    "critical": {"color": SOLARIZED["red"], "bold": True},
}

# The 'verbosity' problem option is scipy's least_squares convention, where 0
# is silent, 1 reports termination and 2 reports every iteration, so 2 is the
# pyCamSet default and has to stay at INFO. 3 is a pyCamSet extension that
# also shows debug records.
VERBOSITY_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.INFO,
    3: logging.DEBUG,
}


def _has_real_handler(logger: logging.Logger) -> bool:
    """
    Whether a logger has a handler that will actually emit something.

    :param logger: the logger to inspect
    """
    return any(
        not isinstance(handler, logging.NullHandler)
        for handler in logger.handlers
    )


def setup_logging(level: int | str = logging.INFO,
                  force: bool = False) -> logging.Logger:
    """
    Install pyCamSet's coloured handler on the ``pyCamSet`` logger.

    Does nothing when logging is already configured, either on the pyCamSet
    logger or on the root logger, so that an application's own configuration
    wins and its output is not duplicated. Pass ``force`` to install over the
    top of an existing pyCamSet handler.

    Propagation is deliberately left on, so records still reach a root handler
    (and pytest's caplog) after this has run.

    :param level: the level to show, as a logging constant or its name
    :param force: replace a handler pyCamSet has already installed
    :return: the pyCamSet logger
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not force and (_has_real_handler(logger)
                      or _has_real_handler(logging.getLogger())):
        return logger

    for handler in [h for h in logger.handlers
                    if not isinstance(h, logging.NullHandler)]:
        logger.removeHandler(handler)

    coloredlogs.install(
        level=level,
        logger=logger,
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        field_styles=LOG_FIELD_STYLES,
        level_styles=LOG_LEVEL_STYLES,
    )
    return logger


def setup_logging_from_verbosity(verbosity: int) -> logging.Logger:
    """
    Install the handler at the level a problem's 'verbosity' option asks for.

    :param verbosity: the 'verbosity' problem option
    :return: the pyCamSet logger
    """
    return setup_logging(VERBOSITY_LEVELS.get(verbosity, logging.INFO))
