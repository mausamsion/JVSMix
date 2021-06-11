from LibriMix.scripts.create_librimix_from_metadata import transform_sources
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
n_cpu = multiprocessing.cpu_count()

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


def process_metadata_file(md_file_path, freqs, n_src, jvs_dir,
                          jvsmix_out_dir, modes, types):
    md_df = pd.read_csv(md_file_path, engine='python')
    # Make a directory for each frequency
    for freq in freqs:
        freq_path = os.path.join(jvsmix_out_dir, 'wav_'+freq)
        # Get actual frequency number (in Hz)
        freq = int(freq.strip('k')) * 1000
        
        # Make directory for each mode inside each frequency directory
        for mode in modes:
            mode_path = os.path.join(freq_path, mode)
            subset_md_path = os.path.join(mode_path, 'metadata')
            os.makedirs(subset_md_path, exist_ok=True)
            # Directory to store actual sources and mixtures
            data_dir_name = os.path.basename(md_file_path).replace(
                                f'jvs{n_src}mix_', '').replace(
                                    '.csv', '')
            dir_path = os.path.join(mode_path, data_dir_name)
            # If files already exist, do not overwrite
            if os.path.isdir(dir_path):
                print('Target dir exist. No overwriting will be done.')
                continue
            print(f'Creating mixutres and sources from {md_file_path} \
                   in {dir_path}')
            if types == ['mix_clean']:
                subdirs = [f's{i+1}' for i in range(n_src)] + ['mix_clean']
            else:
                subdirs = [f's{i+1}' for i in range(n_src)] + types + ['noise']
            # Create subdirectories
            for subdir in subdirs:
                os.makedirs(os.path.join(dir_path, subdir))
            # Process metadata file
            process_metadata_file(md_df, jvs_dir, freq, mode, subdirs, 
                                  dir_path, subset_md_path, n_src)


def process_metadata_file(md_df, jvs_dir, freq, mode, subdirs, 
                          dir_path, subset_md_path, n_src):
    md_dict = {}
    dir_name = os.path.basename(dir_path)
    for subdir in subdirs:
        if subdir.startswith('mix'):
            md_dict[f'metrics_{dir_name}_{subdir}'] = \
                create_empty_md(n_src, subdir, 'metrics')
            md_dict[f'mixture_{dir_name}_{subdir}'] = \
                create_empty_md(n_src, subdir, 'mixture')

    # Multiprocess and get results in dataframes
    chunk_size = 100
    nrows = md_df.shape[0]
    args = [jvs_dir, freq, mode, subdirs, dir_path, subset_md_path, n_src]
    results = []
    executor = get_reusable_executor(max_workers=int(n_cpu*0.75), timeout=120)
    with tqdm(total=nrows//chunk_size, colour='red') as pbar:
        for _,res in enumerate(executor.map(generate_utterances, 
                [ args + [[ row 
                    for _,row in md_df.iloc[i:i+chunk_size].iterrows() ]] 
                        for i in range(0, nrows, chunk_size) ])):
            results.append(res)
            pbar.update()
    
    # Save the dataframes
    for md_name in md_dict:
        md_dict[md_name].to_csv(os.path.join(subset_md_path, md_name+'.csv'), 
                                index=None)


def create_empty_md(n_src, subdir, type):
    empty_df = pd.DataFrame()
    cols = []
    if type == 'metrics':
        cols.append('mixture_id')
        if subdir == 'mix_clean':
            for i in range(n_src):
                cols.append(f'source_{i+1}_snr')
        elif subdir == 'mix_both':
            for i in range(n_src):
                cols.append(f'source_{i+1}_snr')
            cols.append(f'noise_snr')
        elif subdir == 'mix_single':
            cols.append('source_1_snr')
            cols.append(f'noise_snr')
        empty_df.columns = cols
    elif type == 'mixture':
        cols.append('mixture_id')
        cols.append('mixture_path')
        if subdir == 'mix_clean':
            for i in range(n_src):
                cols.append(f'source_{i+1}_path')
        elif subdir == 'mix_both':
            for i in range(n_src):
                cols.append(f'source_{i+1}_path')
            cols.append(f'noise_path')
        elif subdir == 'mix_single':
            cols.append('source_1_path')
            cols.append(f'noise_path')
        cols.append('length')
        empty_df.columns = cols
    return empty_df


def generate_utterances(jvs_dir, freq, mode, subdirs, dir_path,
                        subset_md_path, n_src, rows):
    res = []
    for row in rows:
        # Get sources and mixture infos
        mix_id, gain_list, sources = read_sources(row, n_src, jvs_dir)
        # Transform sources
        transformed_sources = transform_sources()
        # Write sources and get their paths
        source_path_list = write_sources()


def read_sources(row, n_src, jvs_dir):
    mixture_id  = row['mixture_id']
    path_list = []
    gain_list = []
    for k in row.keys():
        if 'path' in k:
            path_list.append(row[k])
        elif 'gain' in k:
            gain_list.append(row[k])
    sources_list = []
    max_length = 0
    # Read audio files to make the mixture
    for source_path in path_list:
        source_path = os.path.join(jvs_dir, source_path)
        source, _ = sf.read(source_path, dtype='float32')
        # Get max length
        if max_length < len(source):
            max_length = len(source)
        sources_list.append(source)
    return mixture_id, gain_list, sources_list


def transform_sources():
    pass

def write_sources():
    pass


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
