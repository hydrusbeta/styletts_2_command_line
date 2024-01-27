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

# Note: This command line utility uses code taken from
# https://github.com/yl4579/StyleTTS2/blob/9b3dd4b910178088b1496a2f97d099f51c1058bb/Demo/Inference_LJSpeech.ipynb
# and modified by HydrusBeta.

import argparse
import json
import os
from collections import OrderedDict

import librosa
import models
import torch
import torchaudio
import yaml
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from munch import Munch

# This is a utility file I created for my (hydrusbeta) own use, but I decided to throw it onto the Git repository. It
# is provided as-is and may not be very user-friendly. It is never invoked by Hay Say.

# Note: for the "precompute" task, all filenames are expected to be named in the format "Character|Trait.ext" where
# "ext" is the audio extension, like .flac or .mp3.


# Constants
INTERNAL_SAMPLERATE = 24000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parse inputs
parser = argparse.ArgumentParser(prog='StyleTTS2 utilities',
                                 description='Various misc utilities for StyleTTS2')
parser.add_argument('--task',             type=str, required=True)
parser.add_argument('--weights_file',     type=str, required=True)
parser.add_argument('--config_file',      type=str, required=True)
parser.add_argument('--input_directory',  type=str, required=True)
parser.add_argument('--output_directory', type=str, required=True)
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

file_names = os.listdir(args.input_directory)
characters = [file_name.split('|')[0] for file_name in file_names]


nested_dict = dict(dict())
if args.task == "precompute":  # precompute style vectors
    for file_name in file_names:
        character, trait = file_name.split('|')
        trait = os.path.splitext(trait)[0]
        in_file_path = os.path.join(args.input_directory, file_name)
        vector = compute_style(in_file_path)
        if nested_dict.get(character) is None:
            nested_dict[character] = dict()
        nested_dict[character][trait] = vector.tolist()[0]

    json_model = {
        "Model": "",
        "Characters": [{
            "Character": character,
            "Pre-computed Styles": [{
                "Trait": trait,
                "Style Vector": vector
            } for trait, vector in styles_dict.items()]
        } for character, styles_dict in nested_dict.items()]
    }

    out_file_path = os.path.join(args.output_directory, 'out.json')
    with open(out_file_path, 'w') as file:
        file.write(json.dumps(json_model, indent=4))
else:
    print('Unknown task "' + args.task + '"')
