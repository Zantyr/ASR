import logging
import librosa
import numpy as np

from lib.core import Data

logger = logging.getLogger("audiologist.loaders")

def seq_phones_loader(seq_path_template, phones_path_template, batch_size=32):
    LENGTH = 2648
    RECS = 1384 * 9
    TRANSL = 307
    X = np.zeros([RECS, LENGTH, 20], np.float32)
    Y = np.zeros([RECS, TRANSL], np.int16)
    counter = 0
    for i in range(9):
        Xpart = np.load("datasets/clarin-long/data/clarin-mfcc-rec-{}.npy".format(i))
        Ypart = np.load("datasets/clarin-long/data/clarin-mfcc-trans-{}.npy".format(i))
        recs = Xpart.shape[0]
        reclen = Xpart.shape[1]
        translen = Ypart.shape[1]
        X[counter : counter + recs, :reclen, :] = Xpart
        Y[counter : counter + recs, :translen] = Ypart
        counter += recs
    logger.debug((counter, RECS))
    counter //= batch_size
    counter *= batch_size
    X = X[:counter]
    Y = Y[:counter]
    return Data(input_seq=X, phones=Y, input_seq_mean=MEAN, input_seq_std=STD)
