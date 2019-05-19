import keras
import keras.backend as K
import librosa
import scipy
import numpy as np
import tensorflow as tf


def pcen(S, sr=22050, hop_length=512, gain=0.98, bias=2, power=0.5,
         time_constant=0.400, eps=1e-6, b=None, max_size=1, ref=None,
         axis=-1, max_axis=None):
    """
    Adapted from librosa (in case it is impossible to get the newest version)
    License: https://github.com/librosa/librosa/blob/master/LICENSE.md
    """
    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter
        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    if np.issubdtype(S.dtype, np.complexfloating):
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError('Max-filtering cannot be applied to 1-dimensional input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError('Max-filtering a {:d}-dimensional spectrogram '
                                         'requires you to specify max_axis'.format(S.ndim))
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    S_smooth = scipy.signal.lfilter([b], [1, b - 1], ref, axis=axis)

    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    return (S * smooth + bias)**power - bias**power


class PcenFeature:
    def __call__(self, recording):
        spec = librosa.stft(recording, n_fft=512, hop_length=128)
        spec = pcen(spec, sr=16000, hop_length=128)
        return librosa.feature.mfcc(S=spec, sr=16000, hop_length=128, n_mfcc=24).T


class ASR:

    _MEAN, _STD = 0.15655953, 1.3286684
    _model_path = "models/pcen_model.h5"
    _transform = PcenFeature()
    _lang = ['tS', 'dz', 'sil', 'en', 'e', 'I', 'v', 'S', 'i', 'r', 'dzi', 'on', 'f', 'n', 'si', 'o', 't', 'k', 'l', 'ts', 's', 'x', 'z', 'ni', 'b', 'j', 'zi', 'g', 'p', 'dZ', 'Z', 'tsi', 'a', 'd', 'm', 'w', 'u']

    def __init__(self):
        self.mdl = keras.models.load_model(self._model_path, {
            "MEAN": self._MEAN, "STD": self._STD, "tf": tf})
        self.transform = self._transform
        self.lang = self._lang
        self.mdl.predict(np.zeros([1, 1, self.mdl.input_shape[2]]))
    
    def predict(self, X):
        feats = np.stack([self.transform(X)])
        phonemes = self.mdl.predict(feats)
        phonemes = K.ctc_decode(phonemes).eval(session=K.get_session())
        [self.lang[x] for x in phonemes[0]]
        return phonemes
    
    def predict_words(self, X):
        raise NotImplementedError()
        feats = np.stack([self.transform(X)])
        phoneme_likelihoods = self.mdl.predict(feats)
        

