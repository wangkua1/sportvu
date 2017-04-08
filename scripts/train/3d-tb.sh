sportvu_root='/ais/gobi5/wangkua1/projects/sportvu/sportvu'
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2-vid-tb1-25x25.yaml $sportvu_root/model/config/conv3d-2layers-25x25.yaml
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2-vid-tb1-25x25.yaml $sportvu_root/model/config/conv3d-2layers-25x25-bn.yaml
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2-vid-tb1-25x25.yaml $sportvu_root/model/config/conv3d-3layers-25x25.yaml
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2-vid-tb1-25x25.yaml $sportvu_root/model/config/conv3d-3layers-25x25-bn.yaml
