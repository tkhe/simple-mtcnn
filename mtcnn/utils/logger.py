import json
import logging
import sys


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger


def log_json_state(state):
    state = {
        k: '{:.6f}'.format(v) if isinstance(v, float) else v
        for k, v in state.items()
    }
    print('{:s}'.format(json.dumps(state)))
