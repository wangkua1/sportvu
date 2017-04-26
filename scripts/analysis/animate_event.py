from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor,EncDecExtractor
from sportvu.data.loader import BaseLoader, Seq2SeqLoader
from sportvu.vis.Event import Event


f_config = '../../sportvu/data/config/rev3-ed-target-history.yaml'
dataset = BaseDataset(f_config, 0, load_raw=True)
# extractor = BaseExtractor(f_config)
# loader = Seq2SeqLoader(dataset, extractor, 100, fraction_positive=0)
# print ('testing next_batch')
# batch = loader.next()
# for eind, event in enumerate(batch[0]):
#     event.show('/home/wangkua1/Pictures/vis/%i.mp4' % eind)
gameid = dataset.games.keys()[0]
eventid = 0
event = dataset.games[gameid]['events'][eventid]
event_obj = Event(event, gameid)
event_obj.show('/u/wangkua1/Pictures/vis/blahj.mp4')

[(idx, event_obj.player_ids_dict[p.id]) for (idx,p) in enumerate(event_obj.moments[0].players)]