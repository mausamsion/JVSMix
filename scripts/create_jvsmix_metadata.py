import os
import argparse
import random
import itertools
import soundfile as sf
import pyloudnorm as pyln
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# Global parameters
EPS = 1e-10             # secures log and division
MAX_AMP = 0.9           # max amplitude in sources and mixture
RATE = 24000            # JVS has all the sources at 24KHz
MIN_LOUDNESS = -33      # loudness randomized between this min and max
MAX_LOUDNESS = -25

random.seed(673)        # sum([ord(ch) for ch in 'jvsmix'])

def main(args):
    create_jvsmix_metadata(args.jvs_dir, args.jvs_md_file, args.md_dir, 
                           args.jsut_name_file, args.n_src)


def create_jvsmix_metadata(jvs_dir, jvs_md_file, md_dir, jsut_name_file, n_src):
    dataset = f'jvs{n_src}mix'
    try: 
        os.mkdir(os.path.join(md_dir, dataset))
    except: 
        pass
    jvs_md = pd.read_csv(jvs_md_file, engine='python')
    jsut_name = pd.read_csv(jsut_name_file, header=None, engine='python')
    jsut_name = jsut_name.set_index(0).to_dict()[1]
    # Generate speaker combinations
    speaker_combs = make_combs(jvs_md, n_src)
    
    # Create and save empty dataframes as CSVs
    mixture_md_cols = ['mixture_id']
    mixture_info_cols = ['mixture_id']
    for i in range(n_src):
        mixture_md_cols.append(f'source_{i + 1}_path')
        mixture_md_cols.append(f'source_{i + 1}_gain')
        mixture_info_cols.append(f'speaker_{i + 1}_id')
        mixture_info_cols.append(f'speaker_{i + 1}_gender')
    mixture_md = pd.DataFrame(columns=mixture_md_cols)
    mixture_info = pd.DataFrame(columns=mixture_info_cols)
    
    # Create metadata for each subset with all speakers
    for subset in ['parallel', 'non-parallel']:
        clip_counter = 0
        save_path = os.path.join(md_dir, dataset, dataset+'_'+subset)
        mixture_md.to_csv(save_path+'.csv', index=None)
        mixture_info.to_csv(save_path+'_info.csv', index=None)
        
        for comb in tqdm(speaker_combs, total=len(speaker_combs)):
            sources_info, sources_list_max = read_sources(
                                                jvs_md, comb, n_src,
                                                jvs_dir, jsut_name, subset)
            # Compute original loudness and normalized sources
            loudness, _, sources_list_norm = set_loudness(sources_list_max)
            # Create mixture
            mixtures = mix(sources_list_norm)
            renormalize_loudness, did_clip = check_for_clipping(
                                                mixtures, sources_list_norm)
            # Keep track of number of clippings done
            # clip_counter += sum([int(i) for i in did_clip])
            # Compute gain
            gains_list = compute_gain(loudness, renormalize_loudness)
            # Add all the info to dataframe
            mix_md_df, mix_info_df = get_dfs(sources_info, gains_list, n_src)
            # Save interim results to file
            mix_md_df.to_csv(save_path+'.csv', 
                             index=None, header=None, mode='a')
            mix_info_df.to_csv(save_path+'_info.csv',
                               index=None, header=None, mode='a')
        # print(f'Among {len(mixture_md)} mixtures, {clip_counter} clipped.')


def make_combs(jvs_md, n_src):
    speakers = list(set(jvs_md['speaker_id']))
    # As number of speakers are less, we are making all possible speaker
    # pairs instead of randomly sampling them
    speaker_combs = list(itertools.combinations(speakers, n_src))
    return speaker_combs


def read_sources(jvs_md, comb, n_src, jvs_dir, jsut_name, subset):
    subdfs = [jvs_md[ (jvs_md['speaker_id']==comb[i]) & 
                      (jvs_md['subset']==subset) ] for i in range(n_src)]
    # Get sources info
    speaker_id_list = comb
    gender_list = [subdfs[i].iloc[0]['gender'] for i in range(n_src)]

    # Each speaker has multiple utterances
    length_list = [list(subdfs[i]['length']) for i in range(n_src)]
    lengths = list(itertools.product(*length_list))
    path_list = [list(subdfs[i]['path']) for i in range(n_src)]
    paths = list(itertools.product(*path_list))
    
    # Generate mixture ids
    mixture_ids = []
    max_lengths = []
    for pc, leng in zip(paths, lengths):
        mixture_ids.append('-'.join([speaker_id_list[i] + '_' + 
                          jsut_name[os.path.split(pc[i])[1].replace('.wav','')] 
                            for i in range(n_src)]))
        max_lengths.append(max(leng))

    assert(len(lengths) == len(paths) == 
            len(mixture_ids) == len(max_lengths))
    # print(f'\tTotal {len(mixture_ids)} combinations generated.')

    # Read the source and compute info
    sources_list = []
    for j in range(len(paths)):
        sltmp = []
        # Randomly pad the smaller audio to the start, end or equally
        padway = random.randint(0, 2)
        for i in range(n_src):
            s, _ = sf.read(os.path.join(jvs_dir, paths[j][i]), dtype='float32')
            padlen = max_lengths[j] - len(s)
            if padway == 0:
                # pad start
                sltmp.append(np.pad(s, (padlen, 0), mode='constant'))
            elif padway == 1:
                # pad equally
                sltmp.append(
                    np.pad(s, (padlen//2, padlen-(padlen//2)), mode='constant'))
            elif padway == 2:
                # pad end
                sltmp.append(np.pad(s, (0, padlen), mode='constant'))
        sources_list.append(sltmp)

    sources_info = {'mixture_ids': mixture_ids, 
                    'speaker_id_list': speaker_id_list, 
                    'gender_list': gender_list, 
                    'paths': paths}
    return sources_info, sources_list


def set_loudness(sources_list):
    loudness_list = []
    meter = pyln.Meter(RATE)
    target_loudness_list = []
    sources_list_norm = []
    for srcs in sources_list:
        src_list_norm = []
        trg_loudness_list = []
        loudness = []
        for i in range(len(srcs)):
            # Initialize loudness
            loudness.append(meter.integrated_loudness(srcs[i]))
            # Pick a random loudness
            target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
            # Normalize source to target loudness
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                src = pyln.normalize.loudness(srcs[i], loudness[i], 
                                              target_loudness)
            if np.max(np.abs(src)) >= 1:
                src = srcs[i] * MAX_AMP / np.max(np.abs(srcs[i]))
                target_loudness = meter.integrated_loudness(src)
            # Save tmp results
            src_list_norm.append(src)
            trg_loudness_list.append(target_loudness)
        # Save final results
        sources_list_norm.append(src_list_norm)
        target_loudness_list.append(trg_loudness_list)
        loudness_list.append(loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(source_list_norm):
    mixtures = []
    for srcs in source_list_norm:
        mixture_max = np.zeros_like(srcs[0])
        for i in range(len(srcs)):
            mixture_max += srcs[i]
        mixtures.append(mixture_max)
    return mixtures


def check_for_clipping(mixtures, sources_list_norm):
    renormalize_loudness = []
    clips = []
    for mixs, srcs in zip(mixtures, sources_list_norm):
        renorm_loudness = []
        clip = False
        meter = pyln.Meter(RATE)
        # Check for clipping in mixtures
        if np.max(np.abs(mixs)) > MAX_AMP:
            clip = True
            weight = MAX_AMP / np.max(np.abs(mixs))
        else:
            weight = 1
        # Renormalize
        for i in range(len(srcs)):
            new_loudness = meter.integrated_loudness(srcs[i] * weight)
            renorm_loudness.append(new_loudness)
        renormalize_loudness.append(renorm_loudness)
        clips.append(clip)
    return renormalize_loudness, clips


def compute_gain(loudness, renormalize_loudness):
    gains = []
    for loud, renorm_loud in zip(loudness, renormalize_loudness):
        gain = []
        for i in range(len(loud)):
            delta_loudness = renorm_loud[i] - loud[i]
            gain.append(np.power(10.0, delta_loudness / 20.0))
        gains.append(gain)
    return gains


def get_dfs(sources_info, gains_list, n_src):
    mix_df = pd.DataFrame()
    info_df = pd.DataFrame()
    for i in range(len(sources_info['mixture_ids'])):
        row_mixture = [sources_info['mixture_ids'][i]]
        row_info = [sources_info['mixture_ids'][i]]
        for j in range(n_src):
            row_mixture.append(sources_info['paths'][i][j])
            row_mixture.append(gains_list[i][j])
            row_info.append(sources_info['speaker_id_list'][j])
            row_info.append(sources_info['gender_list'][j])
        mix_df = mix_df.append([row_mixture])
        info_df = info_df.append([row_info])
    return mix_df, info_df


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jvs_dir', type=str, required=True,
                        help='Absolute path of the JVS dataset directory')
    parser.add_argument('--jvs_md_file', type=str, required=True,
                        help='Absolute path of the JVS dataset metadata file')
    parser.add_argument('--md_dir', type=str, required=True,
                        help='Absolute path to metadata directory')
    parser.add_argument('--jsut_name_file', type=str, required=True,
                        help='Absolute path of file having JSUT audio naming')
    parser.add_argument('--n_src', type=int, required=True, 
                        help='Number of sources to create the mixture for')
    args = parser.parse_args()
    main(args)
