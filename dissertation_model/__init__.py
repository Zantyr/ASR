"""
Handcrafted + MFCC
CZT model
~~~~~
Tasks:
- DenoisingTask
- FeatureLearnabilityTask
Other:
- PER metric
- WER metric
- add Language models to abstract training

> implement models
> implement "mock" option
> installer

> add Language models everywhere
> prepare dataset download scripts

> FeatureLearnabilityTask
> Task.mock
"""


import fwks.model as model_mod
import fwks.noise_gen
import fwks.stage as stage
import fwks.miscellanea as miscellanea

from fwks.stage import RandomSelectionAdapter

import keras
import numpy as np


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


def CZTModel():
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

##################################################################


def stft_max_melfbank_model():
    return model_mod.AcousticModel([
        stage.WindowedTimeFrequencyFBank(preset="stft", summary="max"),
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
        ########################
        stage.LearnableFilterbank(),
        ########################
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
        stage.Window(512, 128),
        ###
        ###
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


def separated_phonemes_model():
    return model_mod.AcousticModel([
        stage.Window(512, 128),
        stage.MelFilterbank(64),
        stage.MeanStdNormalizer(),
        ###
        stage.CNN2D(channels=16, filter_size=5, depth=2),
        stage.RNN(width=512, depth=2),
        stage.phonemic_map(37),
        ###
        stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
    ], callbacks=[
        keras.callbacks.TerminateOnNaN(),
        miscellanea.StopOnConvergence(4),
    ])
