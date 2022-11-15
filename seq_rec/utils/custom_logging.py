import logging
import logging.config
import os

CUR_DIR = os.path.abspath(os.path.join(__file__, "../.."))
logging.config.fileConfig(f"{CUR_DIR}/conf/logging.ini", disable_existing_loggers=False)

# Set default logging level for seq_rec modules, can be overwritten at run time
logging.getLogger('seq_rec').setLevel(logging.DEBUG)
