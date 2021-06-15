import os
import argparse
import random
import glob
from scipy.signal import resample_poly
import pandas as pd
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
from codetiming import Timer
import multiprocessing
from loky import get_reusable_executor

random.seed(673)        # sum([ord(ch) for ch in 'jvsmix'])

class CreateJVSmixDataset():
    def __init__(self, args):
        self.jvs_dir = args.jvs_dir
        self.jvs_md_file = args.jvs_md_file
        self.jvsmix_md_dir = args.jvsmix_md_dir
        self.jvsmix_out_dir = args.jvsmix_out_dir
        self.n_src = args.n_src
        self.freqs = args.freqs
        self.modes = args.modes
        self.types = args.types

        self.EPS = 1e-10     # secures log and division
        self.RATE = 24000    # JVS has all the sources at 24KHz
        self.n_cpu = multiprocessing.cpu_count()
        self.create_jvsmix()

    @Timer(name='decorator')
    def create_jvsmix(self):
        # Get metadata files
        jvs_md_filelist = [file for file in os.listdir(self.jvsmix_md_dir)
                        if 'info' not in file]
        # Create all parts of librimix
        for md_file in jvs_md_filelist:
            md_file_path = os.path.join(self.jvsmix_md_dir, md_file)
            self.process_metadata_file(md_file_path)

    
    def process_metadata_file(self, md_file_path):
        md_df = pd.read_csv(md_file_path, engine='python')
        # Make a directory for each frequency
        for freq in self.freqs:
            freq_path = os.path.join(self.jvsmix_out_dir, 'wav_'+freq)
            # Get actual frequency number (in Hz)
            freq = int(freq.strip('k')) * 1000

            # Make directory for each mode inside each frequency directory
            for mode in self.modes:
                mode_path = os.path.join(freq_path, mode)
                self.subset_md_path = os.path.join(mode_path, 'metadata')
                os.makedirs(self.subset_md_path, exist_ok=True)
                # Directory to store actual sources and mixtures
                data_dir_name = os.path.basename(md_file_path).replace(
                    f'jvs{self.n_src}mix_', '').replace(
                    '.csv', '')
                self.dir_path = os.path.join(mode_path, data_dir_name)
                # If files already exist, do not overwrite
                if os.path.isdir(self.dir_path):
                    print('Target dir exist. No overwriting will be done.')
                    continue
                print(f'Creating mixutres and sources from {md_file_path} \
                    in {self.dir_path}')
                if self.types == ['mix_clean']:
                    self.subdirs = [f's{i+1}' for i in range(self.n_src)] + \
                              ['mix_clean']
                else:
                    self.subdirs = [f's{i+1}' for i in range(self.n_src)] + \
                              self.types + \
                              ['noise']
                # Create subdirectories
                for subdir in self.subdirs:
                    os.makedirs(os.path.join(self.dir_path, subdir))
                # Process metadata file
                self.multiproc_md_file(md_df, freq, mode)

    
    def multiproc_md_file(self, md_df, freq, mode):
        def create_empty_md(subdir, type):
            empty_df = pd.DataFrame()
            cols = []
            if type == 'metrics':
                cols.append('mixture_id')
                if subdir == 'mix_clean':
                    cols += [f'source_{i+1}_snr' for i in range(self.n_src)]
                elif subdir == 'mix_both':
                    cols += [f'source_{i+1}_snr' for i in range(self.n_src)]
                    cols.append(f'noise_snr')
                elif subdir == 'mix_single':
                    cols.append('source_1_snr')
                    cols.append(f'noise_snr')
                empty_df.columns = cols
            elif type == 'mixture':
                cols.append('mixture_id')
                cols.append('mixture_path')
                if subdir == 'mix_clean':
                    cols += [f'source_{i+1}_path' for i in range(self.n_src)]
                elif subdir == 'mix_both':
                    cols += [f'source_{i+1}_path' for i in range(self.n_src)]
                    cols.append(f'noise_path')
                elif subdir == 'mix_single':
                    cols.append('source_1_path')
                    cols.append(f'noise_path')
                cols.append('length')
                empty_df.columns = cols
            return empty_df
        
        md_dict = {}
        dir_name = os.path.basename(self.dir_path)
        for subdir in self.subdirs:
            if subdir.startswith('mix'):
                md_dict[f'metrics_{dir_name}_{subdir}'] = \
                    create_empty_md(subdir, 'metrics')
                md_dict[f'mixture_{dir_name}_{subdir}'] = \
                    create_empty_md(subdir, 'mixture')

        # Multiprocess and get results in dataframes
        chunks = 100
        nrows = md_df.shape[0]
        proc_args = [freq, mode]
        results = []
        executor = get_reusable_executor(max_workers=int(self.n_cpu*0.75), 
                                         timeout=120)
        with tqdm(total=nrows//chunks, colour='red') as pbar:
            for _, res in enumerate(
                executor.map(self.generate_utterances,
                    [proc_args + [
                        [row for _, row in md_df.iloc[i:i+chunks].iterrows()]]
                        for i in range(0, nrows, chunks)])):
                results.append(res)
                pbar.update()

        # Save the dataframes
        for md_name in md_dict:
            md_dict[md_name].to_csv(
                os.path.join(self.subset_md_path, md_name+'.csv'), index=None)

    
    def generate_utterances(self, freq, mode, rows):
        """ Helper functions
        """
        def read_sources(row):
            mix_id = row['mixture_id']
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
                source_path = os.path.join(self.jvs_dir, source_path)
                source, _ = sf.read(source_path, dtype='float32')
                # Get max length
                if max_length < len(source):
                    max_length = len(source)
                sources_list.append(source)
            return mix_id, gain_list, sources_list

        def transform_sources(sources, freq, mode, gains):
            sources_norm = loudness_normalize(sources, gains)
            sources_resampled = resample_sources(sources_norm, freq)
            sources_reshaped = fit_lengths(sources_resampled, mode)
            return sources_reshaped

        def write_sources(mix_id, transformed_sources, freq):
            source_paths = []
            ex_filename = mix_id + '.wav'
            for src, src_dir in zip(transformed_sources[:self.n_src],
                                    self.subdirs[:self.n_src]):
                save_path = os.path.join(self.dir_path, src_dir, ex_filename)
                abs_save_path = os.path.abspath(save_path)
                sf.write(abs_save_path, src, freq)
                source_paths.append(abs_save_path)
            return source_paths

        def loudness_normalize(sources, gains):
            return [source * gains[i] for i, source in enumerate(sources)]

        def resample_sources(sources_norm, freq):
            return [resample_poly(source, freq, self.RATE)
                    for source in sources_norm]

        def fit_lengths(sources_resampled, mode):
            if mode == 'min':
                target_length = min([len(source) for source in sources_resampled])
                return [source[:target_length] for source in sources_resampled]
            else:
                target_length = max([len(source) for source in sources_resampled])
                return [np.pad(source, (0, target_length - len(source)))
                        for source in sources_resampled]
        
        """ Start of the generate_utterances
        """
        res = []
        for row in rows:
            # Get sources and mixture infos
            mix_id, gains, sources = read_sources(row)
            # Transform sources
            transformed_sources = transform_sources(sources, freq, 
                                                    mode, gains)
            # Write sources and get their paths
            source_path_list = write_sources(mix_id, transformed_sources, freq)

            res.append([mix_id, transformed_sources])

        for subdir in self.subdirs:
            if subdir == 'mix_clean':
                sources_to_mix = transformed_sources[:self.n_src]



    


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
    CreateJVSmixDataset(args)
