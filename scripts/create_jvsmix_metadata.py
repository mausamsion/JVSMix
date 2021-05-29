import os
import argparse
import random
import itertools
import soundfile as sf
import pandas as pd
import tqdm as tqdm

# Global parameters
EPS = 1e-10         # secures log and division
MAX_AMP = 0.9       # max amplitude in sources and mixture
RATE = 24000        # JVS has all the sources at 24KHz
MIN_LOUDNESS = -33      # loudness randomized between this min and max
MAX_LOUDNESS = -25

random.seed(23)

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
    jsut_name = pd.read_csv(jsut_name_file, header=None, engine='Python')
    jsut_name = jsut_name.set_index(0).to_dict()[1]
    # Generate speaker combinations
    speaker_combs = make_combs(jvs_md, n_src)
    # Create metadata for each subset with all speakers
    for subset in ['parallel', 'non-parallel']:
        create_jvs_df(jvs_md, jvs_dir, jsut_name, n_src, subset, speaker_combs)


def make_combs(jvs_md, n_src):
    speakers = list(set(jvs_md['speaker_id']))
    # As number of speakers are less, we are making all possible speaker
    # pairs instead of randomly sampling them
    speaker_combs = list(itertools.combinations(speakers, n_src))
    return speaker_combs


def create_jvs_df(jvs_md, jvs_dir, jsut_name, n_src, subset, speaker_combs):
    mixture_md = pd.DataFrame(columns=['mixture_id'])
    mixture_info = pd.DataFrame(columns=['mixture_id'])
    for i in range(n_src):
        mixture_md[f"source_{i + 1}_path"] = {}
        mixture_md[f"source_{i + 1}_gain"] = {}
        mixture_info[f"speaker_{i + 1}_id"] = {}
        mixture_info[f"speaker_{i + 1}_gender"] = {}
    # Create a row for each combination
    for comb in tqdm(speaker_combs, total=len(speaker_combs)):
        sources_info, sources_list_max = read_sources(jvs_md, comb, n_src, 
                                                      jvs_dir, jsut_name, subset)


def read_sources(jvs_md, comb, n_src, jvs_dir, jsut_name, subset):
    subdfs = [jvs_md[ (jvs_md['speaker_id']==comb[i]) & 
                      (jvs_md['subset']==subset) ] for i in range(n_src)]
    # Get sources info
    speaker_id_list = comb
    gender_list = [subdfs[i].iloc[0]['gender'] for i in range(n_src)]
    duration = [subdfs[i].iloc[0]['duration'] for i in range(n_src)]
    path_list = [subdfs[i].iloc[0]['path'] for i in range(n_src)]
    # Each speaker has multiple utterances
    path_list = [subdfs[i]['path'] for i in range(n_src)]
    



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jvs_dir', type=str, required=True,
                        help='Absolute path of the JVS dataset directory')
    parser.add_argument('--jvs_md_file', type=str, required=True,
                        help='Absolute path of the JVS dataset metadata file')
    parser.add_argument('--md_dir', type=str, required=True,
                        help='Absolute path to metadata directory')
    parser.add_argument('--jsut_name_file', type=int, required=True,
                        help='Absolute path of file having JSUT audio naming')
    parser.add_argument('--n_src', type=int, required=True, 
                        help='Number of sources to create the mixture for')
    args = parser.parse_args()
    main(args)
