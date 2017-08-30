from __future__ import print_function

from sportvu.data import constant, annotation
import os
import pandas as pd



def raw_to_gt_format(pnr_dir, game_dir):
	"""
	Read raw annotation files to convert to output similar to the output from make_raw_from_untrained
	Use for future comparison purposes
	"""
	format_dir = os.path.join(pnr_dir, 'format')
	annotations = annotation.prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir)
	annotations = pd.DataFrame(annotations)
	game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values
	for game_id in game_ids:
		game_annotations = annotations.loc[annotations.gameid == game_id, ['eid', 'gameclock', 'quarter']]
		game_annotations.to_csv('%s/format-raw-%s.csv' % (format_dir, game_id),index=False)


if __name__ == '__main__':
	game_dir = constant.game_dir
	pnr_dir = os.path.join(game_dir, 'pnr-annotations')

	raw_to_gt_format(pnr_dir, game_dir)
