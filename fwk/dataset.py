import hashlib
import numpy as np
import os
import random
import scipy.io.wavfile as sio
from fwk import stage
import tqdm


def get_fnames_from_clarin_corpus(ROOT):
    rec_fnames, trans_fnames = [], []
    for i in [x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))]:
        recordings = os.path.join(ROOT, i, "wav")
        transcripts = os.path.join(ROOT, i, "lab")
        for fname in os.listdir(recordings):
            core, extension = fname.split(".")
            assert extension == "wav"
            if os.path.isfile(os.path.join(transcripts, core + ".hlb")):
                rec_fnames.append(os.path.join(recordings, fname))
                trans_fnames.append(os.path.join(transcripts, core + ".hlb"))
    return rec_fnames, trans_fnames

def get_recording_lengths(fnames):
    lens = []
    print("Getting dataset lengths")
    for f in tqdm.tqdm(fnames):
        sr, data = sio.read(f)
        assert sr == 16000
        assert len(data.shape) == 1
        lens.append(len(data))
    return lens

def get_phones_clarin(fname):
    with open(fname, "r", encoding="utf-8") as f:
        s = f.read()
        s = s.split("Phoneme Phoneme")[1]
        s = s.split("\n\n")[0]
        s = [x.split(' ')[1] for x in s.split('\n') if x.strip()]
        s = [x.split('_')[0] for x in s]
    return s


class Dataset:
    
    _accepted_requirements = ["clean", "transcripts", "noisy", "stft"]
    
    def __init__(self, noise_gen=None):
        self.root = None
        self.all_phones = None
        self.recording_lengths = None
        self.rec_fnames = None
        self.trans_fnames = None
        self.transcriptions = None
        self.transcription_lens = None
        self.clean = None
        self.clean_lens = None
        self.noisy = None
        self.noisy_lens = None
        self._hash = None
        self.noise_gen = noise_gen

    def get_from(self, root):
        self.root = root
        rec_fnames, trans_fnames = get_fnames_from_clarin_corpus(self.root)
        lens = get_recording_lengths(rec_fnames)
        sorted_records = np.array(sorted(list(range(len(lens))), key=lambda ix: lens[ix]))
        rec_fnames = [rec_fnames[x] for x in sorted_records]
        trans_fnames = [trans_fnames[x] for x in sorted_records]
        lens = [lens[x] for x in sorted_records]
        self.rec_fnames = rec_fnames
        self.trans_fnames = trans_fnames
        self.recording_lengths = lens       
        
    def generate(self, mapping, requirements):
        assert all([x in self._accepted_requirements for x in requirements])
        if "clean" in requirements:
            self._get_cleans(mapping)
        if "transcripts" in requirements:
            self._get_transcripts()
        if "noisy" in requirements:
            self._get_noisy(mapping)
        if "stft" in requirements:
            self._get_stft()
        self._hash = hashlib.sha512(str(self.rec_fnames).encode("utf-8")).digest().hex()[:16]
            
    def select_first(self, count):
        self.rec_fnames = self.rec_fnames[:count]
        self.trans_fnames = self.trans_fnames[:count]
        self.recording_lengths = self.recording_lengths[:count]

    def select_random(self, count):
        selection = random.sample(range(len(self.rec_fnames)), count)
        self.rec_fnames = [self.rec_fnames[x] for x in selection]
        self.trans_fnames = [self.trans_fnames[x] for x in selection]
        self.recording_lengths = [self.recording_lengths[x] for x in selection]     
        
    def generate_dtype(self, mapping):
        dtype = stage.DType("Array", [max(self.recording_lengths)], np.float32)
        return mapping.output_dtype(dtype)

    @property
    def signature(self):
        return self._hash
    
    def cache_choice(self):
        pass

    def _get_transcripts(self):
        print("Getting list of phones")
        transes = (get_phones_clarin(x) for x in tqdm.tqdm(self.trans_fnames))
        self.all_phones = set()
        def len_add_set(trans):
            self.all_phones |= set(trans)
            return len(trans)
        trans_lens = [len_add_set(x) for x in transes]
        self.all_phones = list(self.all_phones)
        phone_zero = len(self.all_phones)
        print("Getting transcriptions")
        phones_shape = [len(trans_lens), max([x for x in trans_lens]) + 1]
        transes = np.full(phones_shape, phone_zero, np.uint16)
        for num, fname in tqdm.tqdm(enumerate(self.trans_fnames)):
            trans = get_phones_clarin(fname)
            trans = np.array([self.all_phones.index(x) for x in trans])
            transes[num, :len(trans)] = trans
        self.transcriptions = transes
        self.transcription_lens = np.array(trans_lens)

    def _get_cleans(self, mapping):
        print("Getting clean recordings")
        n_recs = len(self.rec_fnames)
        dtype = mapping.output_dtype(stage.DType("Array", [max(self.recording_lengths)], np.float32))
        recordings = np.zeros([n_recs] + dtype.shape, dtype.dtype)
        lens = []
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = sio.read(fname)
            assert sr == 16000
            data = data.astype(np.float32) / 2**15
            data = mapping.map(data)
            lens.append(data.shape[0])
            key = [ix] + [slice(None, x, None) for x in data.shape]
            recordings.__setitem__(key, data)
        try:
            pass # normalize
        except:
            pass # no normalization
        self.clean = recordings
        self.clean_lens = np.array(lens)

    def _get_noisy(self, mapping):
        print("Getting distorted recordings")
        n_recs = len(self.rec_fnames)
        dtype = mapping.output_dtype(stage.DType("Array", [max(self.recording_lengths)], np.float32))
        recordings = np.zeros([n_recs] + dtype.shape, dtype.dtype)
        lens = []
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = sio.read(fname)
            assert sr == 16000
            data = data.astype(np.float32) / 2**15
            data = self.noise_gen.pre(data)
            data = mapping.map(data)
            data = self.noise_gen.post(data)
            lens.append(data.shape[0])
            key = [ix] + [slice(None, x, None) for x in data.shape]
            recordings.__setitem__(key, data)
        try:
            pass  # normalize
        except:
            pass  # no normalization
        self.noisy = recordings
        self.noisy_lens = np.array(lens)
        
    def get_metrics(self, print=True):
        pass
    
    def calculate_metric(self, recording, transcription):
        pass

    def _get_stft(self):
        print("Getting standard STFT")
        n_recs = len(self.rec_fnames)
        frames = max(self.recording_lengths) // 128 - 3
        recordings = np.zeros([n_recs, frames, 257], np.float32)
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = sio.read(fname)
            assert sr == 16000
            data = data.astype(np.float32) / 2**15
            for time in range(self.recording_lengths[ix] // 128 - 3):
                recordings[ix, time, :] = np.log(np.fft.rfft(data[time * 128 : time * 128 + 512]) ** 2 + 2e-12)
        self.stft = recordings


