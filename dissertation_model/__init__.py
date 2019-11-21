import fwks.model as model_mod
import fwks.noise_gen
import fwks.stage as stage
import fwks.miscellanea as miscellanea

from fwks.stage import RandomSelectionAdapter

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


def mfcc_model(preprocessing_distortion=None):
    return model_mod.AcousticModel(
        ([preprocessing_distortion] if preprocessing_distortion \
            is not None else []) + [
            stage.Window(512, 128),
            stage.LogPowerFourier(),
            stage.MelFilterbank(64),
            stage.DCT(24),
            stage.MeanStdNormalizer(),
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            stage.RNN(width=512, depth=2),
            stage.phonemic_map(37),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])


def handcrafted_model(preprocessing_distortion=None):
    return model_mod.AcousticModel(
        ([preprocessing_distortion] if preprocessing_distortion \
        is not None else []) + [
            stage.Window(512, 128),  # is it correct?
            stage.HandCrafted(),
            stage.MeanStdNormalizer(),
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            stage.RNN(width=512, depth=2),
            stage.phonemic_map(37),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])


def plc_model(preprocessing_distortion=None):
    return model_mod.AcousticModel(
        ([preprocessing_distortion] if preprocessing_distortion \
        is not None else []) + [
            stage.Window(512, 128),
            stage.PLC(24),
            stage.MeanStdNormalizer(),
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            stage.RNN(width=512, depth=2),
            stage.phonemic_map(37),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])


def melfilterbank(preprocessing_distortion=None):
    return model_mod.AcousticModel(
        ([preprocessing_distortion] if preprocessing_distortion \
        is not None else []) + [
            stage.Window(512, 128),
            stage.MelFilterbank(64),
            stage.MeanStdNormalizer(),
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            stage.RNN(width=512, depth=2),
            stage.phonemic_map(37),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])


def cochleagram(preprocessing_distortion=None):
    return model_mod.AcousticModel(
        ([preprocessing_distortion] if preprocessing_distortion \
        is not None else []) + [
            stage.Cochleagram(36, 512, 128),
            stage.LogPower(),
            stage.MeanStdNormalizer(),
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            stage.RNN(width=512, depth=2),
            stage.phonemic_map(37),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])


def MFCC_PCEN():
    return model_mod.AcousticModel([
        stage.Window(512, 128),
        stage.LogPowerFourier(),  # Frequency analysis in cochlea
        stage.PCENScaling(),
        stage.MelFilterbank(64),
        stage.DCT(24),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])


def MFCCEqA():
    return model_mod.AcousticModel([
        stage.EqualLoudnessWeighting("A"),
        stage.Window(512, 128),
        stage.LogPowerFourier(),  # Frequency analysis in cochlea
        stage.MelFilterbank(64),
        stage.DCT(24),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])


def MFCCEqB():
    return model_mod.AcousticModel([
        stage.EqualLoudnessWeighting("B"),
        stage.Window(512, 128),
        stage.LogPowerFourier(),  # Frequency analysis in cochlea
        stage.MelFilterbank(64),
        stage.DCT(24),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])


def HandcraftedMFCC():
    return model_mod.AcousticModel([
        stage.Window(512, 128),
        stage.ComposeFeatures([[
                stage.LogPowerFourier(),
                stage.MelFilterbank(64),
                stage.DCT(24),
            ],
            stage.HandCrafted()
        ]),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])


def czt_model():
    return model_mod.AcousticModel([
        stage.Window(512, 128),
        stage.CZT(z=0.999, w=0.9995 * np.exp(2j * np.pi / 512)),
        stage.DCT(24),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])

def stft_max_melfbank_model():
    return model_mod.AcousticModel([
        stage.WindowedTimeFrequencyFBank(),
        stage.LogPower(),
        stage.MelFilterbank(64),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])
    
def learnable_filters_model():
    return model_mod.AcousticModel([
        stage.Window(512, 128),
        stage.PlainPowerFourier(),
        stage.LearnableFourierFBanks(64),
        # stage.DCT(24),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])

def missing_fundamental_model():
    return model_mod.AcousticModel([
        stage.CustomCochleagram(64, 512, 128),
        stage.LogPower(),
        stage.MeanStdNormalizer(),
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])

def separated_phonemes_model_factory(phone_list):
    def maker():
        mapping = [
            ["a", 'o', 'e', 'I', 'i', 'u', 'j', 'w', 'sil'],
            ["tsi", 'ts', 'dz', 'dZ', 'tS', 'dzi', 'S', 'zi', 's', 'z', 'Z', 'si'],
            ['p', 't', 'k', 'b', 'd', 'g', 'r', 'l', 'v', 'f', 'x'],
            ['n', 'on', 'en', 'm', 'ni']
        ]

        def mapping_fn(model, columns):
            print(model.symbol_map, mapping)
            sel = [(
                ([ix for ix, col in enumerate(mapping) if x in col][0]),
                ([col for ix, col in enumerate(mapping) if x in col][0].index(x))
            ) for x in model.symbol_map]
            print(sel)
            sel = [columns[x][:, :, y] for (x, y) in sel]
            return K.stack(sel, axis=-1)

        def make_short_mapping(howmuch):
            def inner(inp):
                return keras.layers.Dense(howmuch)(inp)
            return inner

        def make_snd_mapping(inp):
            return keras.layers.Lambda(lambda x:
                K.softmax(
                    tf.pad(x, [(0, 0), (0, 0), (0, 1)], constant_values=1)
                )
            )(inp)

        concat = stage.Columns([
                [
                    stage.CNN2D(channels=4, filter_size=5, depth=2),
                    stage.RNN(width=256, depth=2),
                    stage.CustomNeural(
                        make_short_mapping(len(x))
                    )
                ]
                for x in mapping
            ], mapping=mapping_fn)

        mdl = model_mod.AcousticModel([
            stage.Window(512, 128),
            stage.LogPowerFourier(),
            stage.MelFilterbank(64),
            stage.DCT(24),
            stage.MeanStdNormalizer(),
            concat,
            stage.CustomNeural(
                make_snd_mapping
            ),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ], callbacks=[
            keras.callbacks.TerminateOnNaN(),
            miscellanea.StopOnConvergence(4),
        ])

        concat.model_ref = mdl
        mdl.symbol_map = phone_list

        return mdl
    return maker