
import optparse
import os
import json
from collections import OrderedDict

def parse_db_address(cfg):
    
    db_address = os.getenv('SPEARMINT_DB_ADDRESS')
    if db_address is None:
        if 'database' in cfg and 'address' in cfg['database']:
            db_address = cfg['database']['address']
        else:
            db_address = 'localhost'

    return db_address
    
parser = optparse.OptionParser(usage="usage: %prog [options] directory")

parser.add_option("--config", dest="config_file",
                  help="Configuration file name.",
                  type="string", default="config.json")

(commandline_kwargs, args) = parser.parse_args()

# Read in the config file
expt_dir  = os.path.realpath(os.path.expanduser(args[0]))
if not os.path.isdir(expt_dir):
    raise Exception("Cannot find directory %s" % expt_dir)
expt_file = os.path.join(expt_dir, commandline_kwargs.config_file)

# try:
with open(expt_file, 'r') as f:
    options = json.load(f, object_pairs_hook=OrderedDict)
# except:
#     raise Exception("config.json did not load properly. Perhaps a spurious comma?")
options["config"]  = commandline_kwargs.config_file


# Set sensible defaults for options
options['chooser']  = options.get('chooser', 'default_chooser')
if 'tasks' not in options:
    options['tasks'] = {'main' : {'type' : 'OBJECTIVE', 'likelihood' : options.get('likelihood', 'GAUSSIAN')}}

# Set DB address
db_address = parse_db_address(options)
if 'database' not in options:
    options['database'] = {'name': 'spearmint', 'address': db_address}
else:
    options['database']['address'] = db_address

if not os.path.exists(expt_dir):
    sys.stderr.write("Cannot find experiment directory '%s'. "
                     "Aborting.\n" % (expt_dir))
    sys.exit(-1)
