sportvu_root='/ais/gobi5/wangkua1/projects/sportvu/sportvu'
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2.yaml $sportvu_root/model/config/conv2d-3layers.yaml
python $sportvu_root/train.py 0 $sportvu_root/data/config/rev2.yaml $sportvu_root/model/config/conv2d-3layers-bn.yaml
