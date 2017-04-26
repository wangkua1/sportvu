
from sportvu.data.utils import pictorialize_fast
import numpy as np

def make_sequence_sample_image(history_sequences, future_target_sequences, predicted_sequences, target_index):
    """
    similar to make_sequence_prediction_image,
    but predicted_sequences are samples for the same instance

    """
    kwargs = {'sample_rate':1, 'Y_RANGE':100, 'X_RANGE':50, 'keep_channels':False}
    ## (B, 11, T, 2)
    context_vid = pictorialize_fast(history_sequences, **kwargs)
    context_img = np.sum(context_vid, axis=2)
    context_img[context_img>1] = 1 #(B,3,x,y)

    def _pictorialize_single_sequence(seq, kwargs):
        shape = list(seq.shape) #(B, T, 2)
        shape.insert(1,11)
        tmp = np.zeros((shape))
        tmp[:,0]= seq
        tmp_img = pictorialize_fast(tmp, **kwargs)
        tmp_img = np.sum(tmp_img[:,0],axis=1) # (B, X, Y)
        tmp_img[tmp_img>1] = 1
        return tmp_img
    pred_img = np.zeros(context_img.shape)
    pred_img[:,0] =_pictorialize_single_sequence( history_sequences[:,target_index], kwargs)
    pred_img[:,1] =_pictorialize_single_sequence( future_target_sequences, kwargs)
    tmp = _pictorialize_single_sequence( predicted_sequences, kwargs)
    tmp = tmp.sum(0)
    tmp = tmp / tmp.max()
    pred_img[:,2] =tmp[None]
    ###
    final_img = np.concatenate([context_img, pred_img], axis=-1)
    return np.transpose(final_img, (0,2,3,1))

def make_sequence_prediction_image(history_sequences, future_target_sequences, predicted_sequences, target_index):
    """

    2 concatenated images: 1 for history/context (Ball, Offense, Defense) only up to pre-prediction
                           1 for target          (History, GT, prediction)
    """
    kwargs = {'sample_rate':1, 'Y_RANGE':100, 'X_RANGE':50, 'keep_channels':False}
    ## (B, 11, T, 2)
    context_vid = pictorialize_fast(history_sequences, **kwargs)
    context_img = np.sum(context_vid, axis=2)
    context_img[context_img>1] = 1 #(B,3,x,y)

    def _pictorialize_single_sequence(seq, kwargs):
        shape = list(seq.shape) #(B, T, 2)
        shape.insert(1,11)
        tmp = np.zeros((shape))
        tmp[:,0]= seq
        tmp_img = pictorialize_fast(tmp, **kwargs)
        tmp_img = np.sum(tmp_img[:,0],axis=1) # (B, X, Y)
        tmp_img[tmp_img>1] = 1
        return tmp_img
    pred_img = np.zeros(context_img.shape)
    pred_img[:,0] =_pictorialize_single_sequence( history_sequences[:,target_index], kwargs)
    pred_img[:,1] =_pictorialize_single_sequence( future_target_sequences, kwargs)
    pred_img[:,2] =_pictorialize_single_sequence( predicted_sequences, kwargs)
    ###
    final_img = np.concatenate([context_img, pred_img], axis=-1)
    return np.transpose(final_img, (0,2,3,1))

if __name__ == '__main__':
    f_data_config = 'data/config/rev3-ed-target-history.yaml'
    # data
    from sportvu.data.dataset import BaseDataset
    from sportvu.data.extractor import Seq2SeqExtractor, EncDecExtractor
    from sportvu.data.loader import Seq2SeqLoader
    import yaml
    from utils import experpolate_position
    # Initialize dataset/loader
    data_config = yaml.load(open(f_data_config, 'rb'))
    dataset = BaseDataset(data_config, 0 , load_raw=True)
    extractor = eval(data_config['extractor_class'])(data_config)
    nfh = 0

    loader = Seq2SeqLoader(dataset, extractor, data_config[
        'batch_size'], fraction_positive=0.5,
        negative_fraction_hard=nfh)

    cloader = loader

    loaded = cloader.next()
    if loaded is not None:
        dec_input, dec_output, enc_input, meta = loaded
    else:
        cloader.reset()
    history, pid = meta

    gt_future = experpolate_position(history[:,pid,-1], dec_output)
    img = make_sequence_prediction_image(history, gt_future, gt_future, pid)
    import matplotlib.pylab as plt
    plt.ioff()
    f = plt.figure()
    for i in xrange(len(gt_future)):
        plt.imshow(img[i])
        plt.savefig('test-vis-%g.png'%i)