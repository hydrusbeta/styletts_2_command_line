#     StyleTTS2 Command Line Interface
#     Copyright (c) 2023 Aaron (Yinghao) Li
#     Copyright (c) 2024 HydrusBeta
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Note: The original script by Aaron Li was taken from
# https://github.com/yl4579/StyleTTS2/blob/9b3dd4b910178088b1496a2f97d099f51c1058bb/Demo/Inference_LJSpeech.ipynb and
# modified by HydrusBeta.

import argparse
from collections import OrderedDict

import gruut
import librosa
import models
import nltk.data
import numpy as np
import soundfile
import torch
import torchaudio
import yaml
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from gruut.const import Word
from munch import Munch
from nltk.tokenize import word_tokenize
from text_utils import TextCleaner

# Constants
INTERNAL_SAMPLERATE = 24000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parse inputs
parser = argparse.ArgumentParser(prog='StyleTTS2',
                                 description='A Text-to-speech framework that uses style diffusion and adversarial '
                                             'training with large speech language models')
parser.add_argument('-t', '--text',              type=str,                       required=True)
parser.add_argument('-w', '--weights_file',      type=str,                       required=True)
parser.add_argument('-c', '--config_file',       type=str,                       required=True)
parser.add_argument('-o', '--output_filepath',   type=str,                       required=True)
parser.add_argument('-n', '--noise',             type=float, default=0.3)
parser.add_argument('-s', '--style_blend',       type=float, default=0.5)
parser.add_argument('-d', '--diffusion_steps',   type=int,   default=10)
parser.add_argument('-e', '--embedding_scale',   type=float, default=1.0)
parser.add_argument('-l', '--use_long_form',     action='store_true', default=False)
parser.add_argument('-i', '--reference_audio',   type=str,   default=None)
parser.add_argument('-r', '--timbre_ref_blend',  type=float, default=0.3)
parser.add_argument('-p', '--prosody_ref_blend', type=float, default=0.1)
args = parser.parse_args()

# Load pretrained ASR model
config = yaml.safe_load(open(args.config_file))
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = models.load_ASR_models(ASR_path, ASR_config)

# Load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = models.load_F0_models(F0_path)

# Load BERT model
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

# define helper methods
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path):
    wave, sr = librosa.load(path)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != INTERNAL_SAMPLERATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=INTERNAL_SAMPLERATE)
    mel_tensor = preprocess(audio).to(DEVICE)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def convert_text_to_ipa(text: str):
    sentences = gruut.sentences(text)
    ipa_words = [convert_word_to_ipa(word) for sentence in sentences for word in sentence]
    ipa_text = ' '.join(ipa_words)
    return ipa_text


def convert_word_to_ipa(word: Word):
    if not word.is_spoken:  # e.g. punctuation marks
        ipa_word = word.text
    elif word.phonemes:
        ipa_word = ''.join(word.phonemes)
    else:
        ipa_word = ''
    return ipa_word


# Build model
model_params = recursive_munch(config['model_params'])
model = models.build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(DEVICE) for key in model]
params_whole = torch.load(args.weights_file, map_location=DEVICE)
params = params_whole['net']
for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

# Define main inference method
textcleaner = TextCleaner()


def infer(text, s_previous, scaled_noise, diffusion_steps=5, embedding_scale=1, ref_s=None, alpha=0.25, beta=0.25, t=0.7):
    text = text.strip()
    text = text.replace('"', '')
    ps = [convert_text_to_ipa(text)]
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = sampler(scaled_noise,
                         embedding=bert_dur,
                         num_steps=diffusion_steps,
                         embedding_scale=embedding_scale,
                         features=ref_s).squeeze(0)
        if s_previous is not None:
            # convex combination of previous and current style
            s_pred = t * s_previous + (1 - t) * s_pred
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        if ref_s is not None:
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            s_pred = torch.cat([ref, s], dim=-1)
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(DEVICE))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new
        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(DEVICE))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new
        out = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))
    return out.squeeze().cpu().numpy()[..., :-100], s_pred  # remove pulse artifact at end


# Perform inference
noise = args.noise * torch.randn(1, 1, 256).to(DEVICE)
s_ref = compute_style(args.reference_audio) if args.reference_audio is not None else None

if args.use_long_form:
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(args.text)
    audio_outputs = []
    s_prev = None
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        audio_output, s_prev = infer(text=sentence,
                                     s_previous=s_prev,
                                     scaled_noise=noise,
                                     diffusion_steps=args.diffusion_steps,
                                     embedding_scale=args.embedding_scale,
                                     ref_s=s_ref,
                                     alpha=args.timbre_ref_blend,
                                     beta=args.prosody_ref_blend,
                                     t=args.style_blend,)
        audio_outputs.append(audio_output)
    audio_output = np.concatenate(audio_outputs).ravel()
else:
    audio_output, s_prev = infer(text=args.text,
                                 s_previous=None,
                                 scaled_noise=noise,
                                 diffusion_steps=args.diffusion_steps,
                                 embedding_scale=args.embedding_scale,
                                 ref_s=s_ref,
                                 alpha=args.timbre_ref_blend,
                                 beta=args.prosody_ref_blend)

# Write the output file
soundfile.write(args.output_filepath, audio_output, INTERNAL_SAMPLERATE, format='FLAC')

