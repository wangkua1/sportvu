import pandas as pd
import math
import cPickle as pickle

def read_annotation(fpath):
    df = pd.read_csv(open(fpath,'rb'), header=None)
    annotations = {}
    for ind, row in df.iterrows():
        anno = []
        for pnr in row[1:]:
            if type(pnr)==str and ':' in pnr:
                m, s = pnr.split(':')
                anno.append(int(m) *60 + int(s))
            elif type(pnr)==str and pnr[0]=='-':
                pass # this is accepted format
            elif type(pnr)==float and math.isnan(pnr):
                pass # this is accepted format
            else:
                print pnr, pnr=='-'
                raise Exception('unknown annotation format')
        annotations[row[0]] = anno
    return annotations

def read_annotation_from_raw(fpath, game_id):
    data = pickle.load(open(fpath, 'rb'))
    annotations = {}
    for ind, row in enumerate(data):
        anno = []
        if row['gameid'] == str(game_id):
            anno.append(row['gameclock'])
        annotations[row['eid']] = anno
    return annotations

def prepare_gt_file_from_raw_label_dir(pnr_dir):
    gt = []
    for pnr_anno_ind in xrange(len(os.listdir(pnr_dir))):
        game_anno_base = os.listdir(pnr_dir)[pnr_anno_ind]
        if not os.path.isfile(os.path.join(pnr_dir,game_anno_base)):
            continue
        game_id = game_anno_base.split('.')[0].split('-')[1]
        with open(os.path.join(game_dir, game_id+'.pkl'),'rb') as f:
            raw_data = pickle.load(f)
        fpath = os.path.join(pnr_dir, game_anno_base)
        anno = read_annotation(fpath)
        for k, v in anno.items():
            if len(v) == 0:
                continue
            gt_entries = []
            q = raw_data['events'][k]['quarter']
            for vi in v:
                gt_entries.append({'gameid':game_id, 'quarter':q, 'gameclock':vi, 'eid':k})
            gt += gt_entries
    return gt


def script_anno_rev0():
    gt = prepare_gt_file_from_raw_label_dir(pnr_dir)
    pickle.dump(gt, open(os.path.join(pnr_dir,'gt/rev0.pkl'),'wb'))

if __name__ == '__main__':
    from sportvu import data
    from sportvu.vis.Game import Game
    from sportvu.vis.Event import Event
    import os
    game_dir = data.constant.game_dir
    pnr_dir = os.path.join(data.constant.data_dir, 'data/pnr-annotations')
    script_anno_rev0()
