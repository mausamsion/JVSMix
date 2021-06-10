from LibriMix.scripts.create_librimix_from_metadata import process_metadata_file
import os
import argparse
import random
import glob
import itertools
import pandas as pd
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
from codetiming import Timer
import multiprocessing
from loky import get_reusable_executor

# Global parameters
EPS = 1e-10     # secures log and division
RATE = 24000    # JVS has all the sources at 24KHz

random.seed(673)        # sum([ord(ch) for ch in 'jvsmix'])

def main(args):
    create_jvsmix(args.jvs_dir, args.jvs_md_file, args.jvsmix_md_dir, 
                  args.jvsmix_out_dir, args.n_src, args.freqs, 
                  args.modes, args.types)


def create_jvsmix(jvs_dir, jvs_md_file, jvsmix_md_dir, jvsmix_out_dir, n_src, 
                  freqs, modes, types):
    # Get metadata files
    jvs_md_filelist = [file for file in os.listdir(jvsmix_md_dir) 
                                if 'info' not in file]
    # Create all parts of librimix
    for md_file in jvs_md_filelist:
        md_file_path = os.path.join(jvsmix_md_dir, md_file)
        process_metadata_file(md_file_path, freqs, n_src, jvs_dir, 
                              jvsmix_out_dir, modes, types)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jvs_dir', type=str, required=True,
                        help='Absolute path of the JVS dataset directory')
    parser.add_argument('--jvs_md_file', type=str, required=True,
                        help='Absolute path of the JVS dataset metadata file')
    parser.add_argument('--jvsmix_md_dir', type=str, required=True,
                        help='Absolute path to jvsmix metadata directory')
    parser.add_argument('--jvsmix_out_dir', type=str, required=True, 
                        help='Absolute path of directory to output jvsmix data')
    parser.add_argument('--n_src', type=int, required=True,
                        help='Number of sources to create the mixture for')
    parser.add_argument('--freqs', nargs='+', default=['8k', '16k'], 
                        help=('Frequencies of data to generate'
                             '(stored in separate directories)'))
    parser.add_argument('--modes', nargs='+', default=['min', 'max'], 
                        help='Modes to create data for')
    parser.add_argument('--types', nargs='+', 
                        default=['mix_clean', 'mix_both', 'mix_single'], 
                        help='Types of data to generate')
    parser.add_argument('--multiproc', type=bool, default=False,
                        help='Whether to do multiprocessing(default=False)')
    args = parser.parse_args()
    main(args)
