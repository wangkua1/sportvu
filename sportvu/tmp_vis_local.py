

import numpy as np
import cPickle as pkl
import matplotlib.pylab as plt
imgs = pkl.load(open('tmp.pkl','rb'))
imgs= np.array(imgs)[:,0]



for i in xrange(len(imgs)):
	plt.imshow(imgs[i])
	plt.savefig('tmp_imgs/%i.png'%i)
        