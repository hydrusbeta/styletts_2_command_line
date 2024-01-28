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
import json
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
import re
from arpabetandipaconvertor.arpabet2phoneticalphabet import ARPAbet2PhoneticAlphabetConvertor

# Constants
INTERNAL_SAMPLERATE = 24000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parse inputs
parser = argparse.ArgumentParser(prog='StyleTTS2',
                                 description='A Text-to-speech framework that uses style diffusion and adversarial '
                                             'training with large speech language models')
parser.add_argument('-t', '--text',                        type=str,                  required=True)
parser.add_argument('-w', '--weights_file',                type=str,                  required=True)
parser.add_argument('-c', '--config_file',                 type=str,                  required=True)
parser.add_argument('-o', '--output_filepath',             type=str,                  required=True)
parser.add_argument('-n', '--noise',                       type=float, default=0.3)
parser.add_argument('-b', '--style_blend',                 type=float, default=0.5)
parser.add_argument('-d', '--diffusion_steps',             type=int,   default=10)
parser.add_argument('-e', '--embedding_scale',             type=float, default=1.0)
parser.add_argument('-l', '--use_long_form',               action='store_true')
parser.add_argument('-r', '--timbre_ref_blend',            type=float, default=0.3)
parser.add_argument('-p', '--prosody_ref_blend',           type=float, default=0.1)
group = parser.add_mutually_exclusive_group()
group.add_argument('-i', '--reference_audio',              type=str,   default=None)
group.add_argument('-s', '--reference_style_json',         type=str,   default=None)
parser.add_argument('-m', '--precomputed_style_model',     type=str)  # used only with reference_style_json
parser.add_argument('-g', '--precomputed_style_character', type=str)  # used only with reference_style_json
parser.add_argument('-a', '--precomputed_style_trait',     type=str)  # used only with reference_style_json
parser.add_argument('--speed',                             type=float, default=1.0)
args = parser.parse_args()

if args.reference_style_json is None:
    if args.precomputed_style_model or args.precomputed_style_character or args.precomputed_style_trait:
        parser.error('The options --precomputed_style_model --precomputed_style_character and --precomputed_style_trait'
                     ' are not allowed if --reference_style_json is not specified.')
else:  # So, args.reference_style_json is NOT None
    if not (args.precomputed_style_model and args.precomputed_style_character and args.precomputed_style_trait):
        parser.error('If you pass a value for --reference_style_json, then you must also specify values for '
                     '--precomputed_style_model --precomputed_style_character and --precomputed_style_trait')

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


def locate_arpa_ipa_and_plaintext_spans(text: str):
    # locate ARPAbet and IPA spans using regex:
    arpabet_spans = [m.span(0) for m in re.finditer(r"{(.*?)}", text)]
    ipa_spans = [m.span(0) for m in re.finditer(r"<(.*?)>", text)]
    arpa_or_ipa_spans = sorted(arpabet_spans + ipa_spans)

    # locate plaintext spans:
    if arpa_or_ipa_spans:
        # The plaintext is everything in between and outside the ARPAbet and IPA spans:
        plaintext_spans = [(a[1], b[0]) for a, b in zip(arpa_or_ipa_spans[:-1], arpa_or_ipa_spans[1:])]
        plaintext_spans = [(0, arpa_or_ipa_spans[0][0])] + plaintext_spans + [(arpa_or_ipa_spans[-1][-1], len(text))]
    else:
        # edge case - when there are no ARPAbet or IPA substitutions in text
        plaintext_spans = [(0, len(text)-1)]
    # More edge cases - when the very first or last word is an ARPAbet or IPA substitution:
    if (0, 0) in plaintext_spans:
        plaintext_spans.remove((0, 0))
    if (len(text), len(text)) in plaintext_spans:
        plaintext_spans.remove((len(text), len(text)))

    all_spans = sorted(arpa_or_ipa_spans + plaintext_spans)

    return all_spans, arpabet_spans, ipa_spans, plaintext_spans


def convert_text_to_ipa(text: str):
    all_spans, arpabet_spans, ipa_spans, plaintext_spans = locate_arpa_ipa_and_plaintext_spans(text)
    arpa_converter = ARPAbet2PhoneticAlphabetConvertor()

    all_ipa_words = []
    for span in all_spans:
        span_text = (text[span[0]:span[1]]
                     .replace("{", "")
                     .replace("}", "")
                     .replace("<", "")
                     .replace(">", "")
                     .strip())
        if span in ipa_spans:
            all_ipa_words = all_ipa_words + [span_text]
        elif span in arpabet_spans:
            all_ipa_words = all_ipa_words + [arpa_converter.convert_to_international_phonetic_alphabet(span_text)]
        else:
            sentences = gruut.sentences(span_text)
            all_ipa_words = all_ipa_words + [convert_word_to_ipa(word) for sentence in sentences for word in sentence]

    return ' '.join(all_ipa_words)


def convert_word_to_ipa(word: Word):
    if not word.is_spoken:  # e.g. punctuation marks
        ipa_word = word.text
    elif word.phonemes:
        ipa_word = ''.join(word.phonemes)
    else:
        ipa_word = ''
    return ipa_word


def get_precomputed_style_from_file(file_path, model_name, character, trait):
    with open(file_path, 'r') as file:
        json_contents = json.load(file)
        nested_list = {item['Model']: {subitem['Character']: {subsubitem['Trait']: subsubitem['Style Vector']
                                                              for subsubitem in subitem['Pre-computed Styles']}
                                       for subitem in item['Characters']}
                       for item in json_contents}
        style_array = nested_list.get(model_name).get(character).get(trait)
        if style_array is None:
            raise Exception('No style array found for model ' + model_name + ', character ' + character + ' and trait '
                            + trait)
        return torch.FloatTensor([style_array])


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


def infer(text, s_previous, scaled_noise, diffusion_steps=5, embedding_scale=1, ref_s=None, alpha=0.25, beta=0.25,
          t=0.7, speed=1.0):
    text = text.strip()
    text = text.replace('"', '')
    ps = convert_text_to_ipa(text)
    print("IPA Transcription:", flush=True)
    print(ps, flush=True)
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
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
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


# get reference style
if args.reference_audio:
    s_ref = compute_style(args.reference_audio)
elif args.reference_style_json:
    s_ref = get_precomputed_style_from_file(args.reference_style_json,
                                            args.precomputed_style_model,
                                            args.precomputed_style_character,
                                            args.precomputed_style_trait)
else:
    s_ref = None

# Perform inference
noise = args.noise * torch.randn(1, 1, 256).to(DEVICE)

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
                                     t=args.style_blend,
                                     speed=args.speed)
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
                                 beta=args.prosody_ref_blend,
                                 speed=args.speed)

# Write the output file
soundfile.write(args.output_filepath, audio_output, INTERNAL_SAMPLERATE, format='FLAC')

