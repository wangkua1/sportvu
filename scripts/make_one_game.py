"""make_one_game.py

Usage:
    make_one_game.py --list <index> <dir-prefix>
    make_one_game.py --gameid <gameid> <index> <dir-prefix>
    make_one_game.py [--annotate] [--list] <index> <dir-prefix> <pnr-prefix> <time-frame-radius>
    make_one_game.py --annotate [--list] [--gameid] <gameid> <dir-prefix> <pnr-prefix> <time-frame-radius>
    make_one_game.py --annotate --gameid <gameid> <index> <dir-prefix> <pnr-prefix> <time-frame-radius>
    make_one_game.py --from_raw --gameid <gameid> <index> <dir-prefix> <pnr-prefix> <time-frame-radius> <raw_file>

Arguments:
    <index> not a very good way of doing things, this is the index into os.listdir
    <dir-prefix> the prefix prepended the directory that will be created to hold the videos
    <pnr-prefix> the prefix for annotation filenames (e.g. 'raw')
    <time-frame-radius> tfr, let annotated event be T_a, we extract frames [T_a-tfr, T_a+tfr]
    <raw_file> location of annotation file

Options:
    --list: for processing more than one game
    --annotate: use annotation
    --from_raw: use raw predictions to plot animation of pnr

Example:
    python make_one_game.py --from_raw --gameid 0021500408 1 viz raw 75 from-raw-examples.pkl
"""

from sportvu import data
from sportvu.vis.Game import Game
from sportvu.vis.Event import Event, EventException
import os
import cPickle as pickle


from docopt import docopt

game_dir = data.constant.game_dir
pnr_dir = os.path.join(game_dir, 'pnr-annotations')


arguments = docopt(__doc__, version='something 1.1.1')
print ("...Docopt... ")
print(arguments)
print ("............\n")

def wrapper_render_one_game(index, dir_prefix, gameid=None):
    ### Load game
    print ('Loading')
    if gameid != None:
        game_basename = gameid+'.pkl'
    else:
        game_basename = os.listdir(game_dir)[index]

    game_pkl = os.path.join(game_dir, game_basename)
    with open(game_pkl,'rb') as f:
        raw_data = pickle.load(f)
    game_str = "{visitor}@{home}, on {date}".format(visitor = raw_data['events'][0]['visitor']['abbreviation']
                                                  ,home = raw_data['events'][0]['home']['abbreviation']
                                                  ,date=  raw_data['gamedate'] )
    print (game_str)


    ### Create a new directory for videos
    vid_dir =os.path.join(game_dir, 'video') # base dir that holds all the videos
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    new_dir = os.path.join(vid_dir, '{prefix}-{game_id}'.format(
                                                            prefix = dir_prefix
                                                            ,game_id = game_basename.split('.')[0]))
    previous_rendered_events = []
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else: # already a directory exists, likely we've tried to do the same thing
        print(new_dir)
        print('Already exists, not rerunning events rendered and saved previously')
        previous_rendered_events = os.listdir(new_dir)

    render_one_game(raw_data
                , new_dir
                , [int(name.split('.')[0].split('-')[0]) for name in previous_rendered_events ])


def render_one_game(raw_data, directory, skip_these):
    """
    Input:
        raw_data: the huge dictionary of a single game
    """
    N = len(raw_data['events'])
    if arguments['--annotate']:
        pnr_annotations = data.read_annotation(os.path.join(pnr_dir,arguments['<pnr-prefix>']+'-'+raw_data['gameid']+'.csv'))
    elif arguments['--from_raw']:
        pnr_annotations = data.read_annotation_from_raw(os.path.join(pnr_dir, 'gt/%s' % (arguments['<raw_file>'])), raw_data['gameid'])
    for i in xrange(N):
        if i in skip_these:
            print ('Skipping event <%i>'%i)
            continue
        e = Event(raw_data['events'][i])
        ## preprocessing
        if arguments['--annotate'] or arguments['--from_raw']:
            if i not in pnr_annotations.keys():
                print "Clip index %i not labelled"%i
                continue
            for pnr_ind, T_a in enumerate(pnr_annotations[i]):
                e = Event(raw_data['events'][i])
                ## render
                try:
                    e.sequence_around_t(T_a, int(arguments['<time-frame-radius>']))
                    e.show(os.path.join(directory, '%i-pnr-%i.mp4' %(i, pnr_ind)))
                except EventException as e:
                    print ('malformed sequence, skipping')
                    continue
        else:
            ## truncate
            if i < N-1:
                e.truncate_by_following_event(raw_data['events'][i + 1])
            ## render
            try:
                print('Creating video for event: %i' % (i))
                e.show(os.path.join(directory, '%i.mp4' % i))
            except EventException as e:
                print ('malformed sequence, skipping')
                continue

gameid = None
if arguments['--gameid']:
    gameid = arguments['<gameid>']

if arguments['--list']:
    indices = [int(ind) for ind in arguments['<index>'].split(',')]
    dir_prefix = arguments['<dir-prefix>']
    for index in indices:
        wrapper_render_one_game(index, dir_prefix)
else:
    index = arguments['<index>']
    if index != None:
        index = int(index)
    dir_prefix = arguments['<dir-prefix>']
    wrapper_render_one_game(index, dir_prefix, gameid)
