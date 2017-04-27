"""animate_event.py

Usage:
    animate_event.py <f_anim_config>

"""

from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor, EncDecExtractor
from sportvu.data.loader import BaseLoader, Seq2SeqLoader
from sportvu.vis.Event import Event
import cPickle as pkl
import yaml
import os
from docopt import docopt

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")
f_anim_config = arguments['<f_anim_config>']
anim_config = yaml.load(open(f_anim_config, 'rb'))

futures = pkl.load(open(anim_config['tmp_pkl'], 'rb'))

f_config = 'data/config/rev3-ed-target-history.yaml'
dataset = BaseDataset(f_config, 0, load_raw=True)


gameid = anim_config['gameid']
eventid = anim_config['event_id']
event = dataset.games[gameid]['events'][eventid]
event_obj = Event(event, gameid)
event_obj.show('/u/wangkua1/Pictures/vis/%s.mp4' %
               os.path.basename(f_anim_config).split('.')[0], futures)

# [(idx, event_obj.player_ids_dict[p.id]) for (idx,p) in enumerate(event_obj.moments[0].players)]
