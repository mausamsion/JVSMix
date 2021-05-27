import os
import argparse
import random
import itertools
import soundfile as sf
import pandas as pd
import tqdm as tqdm

# Global parameters
EPS = 1e-10         # secures log and division
MAX_AMP = 0.9       # max amplitude in sources and mixtures
RATE = 24000        # JVS has all the sources at 24KHz
MIN_LOUDNESS = -33      # loudness randomized between this min and max
MAX_LOUDNESS = -25

random.seed(23)

def main(args):
    create_jvsmix_metadata(args.jvs_dir, 
                           args.jvs_md_file, 
                           args.md_dir, 
                           args.jsut_name_file, 
                           args.n_src)

def create_jvsmix_metadata(jvs_dir, jvs_md_file, md_dir, jsut_name_file, n_src):
    dataset = f'jvs{n_src}mix'
    try: os.mkdir(os.path.join(md_dir, dataset))
    except: pass
    jvs_md = pd.read_csv(jvs_md_file, engine='python')
    jsut_name = pd.read_csv(jsut_name_file, header=None, engine='Python')
    create_jvs_df(jvs_md, jvs_dir, jsut_name, n_src)

def create_jvs_df(jvs_md, jvs_dir, jsut_name, n_src):
    mixtures_md = pd.DataFrame(columns=['mixture_id'])
    mixtures_info = pd.DataFrame(columns=['mixture_id'])
    for i in range(n_src):
        mixtures_md[f"source_{i + 1}_path"] = {}
        mixtures_md[f"source_{i + 1}_gain"] = {}
        mixtures_info[f"speaker_{i + 1}_id"] = {}
        mixtures_info[f"speaker_{i + 1}_gender"] = {}
    # Generate speaker combinations
    speaker_combs = make_combs(jvs_md, n_src)

def make_combs(jvs_md, n_src):
    speakers = list(set(jvs_md['speaker_id']))
    # As number of speakers are less we are making all possible speaker
    # pairs instead of randomly sampling them
    speaker_combs = list(itertools.combinations(speakers, n_src))
    return speaker_combs



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
