import os
import argparse
import random
import time
import glob
from scipy.signal import resample_poly
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from codetiming import Timer
import multiprocessing
from loky import get_reusable_executor

random.seed(673)        # sum([ord(ch) for ch in 'jvsmix'])

class CreateJVSmixDataset():
    def __init__(self, args):
        self.jvs_dir = args.jvs_dir
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
                print(f'\n-----\nCreating mixutres and sources from '
                      f'{os.path.basename(md_file_path)} '
                      f'in {self.dir_path}\n-----\n')
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
                self.multiproc_md_file(md_file_path, freq, mode)


    def multiproc_md_file(self, md_file_path, freq, mode):
        """ Helper functions
        """
        def create_empty_md(subdir, type):
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
            empty_df = pd.DataFrame(columns=cols)
            return empty_df
        
        """ Start of the func definition
        """
        md_df = pd.read_csv(md_file_path, engine='python')
        md_dict = {}
        dir_name = os.path.basename(self.dir_path)
        for subdir in self.subdirs:
            if subdir.startswith('mix'):
                md_dict[f'metrics_{dir_name}_{subdir}'] = \
                    create_empty_md(subdir, 'metrics')
                md_dict[f'mixture_{dir_name}_{subdir}'] = \
                    create_empty_md(subdir, 'mixture')

        # Tmp dir to save intermediate results from processes
        tmp_md_dir = os.path.join(self.subset_md_path, 'tmp')
        os.makedirs(tmp_md_dir, exist_ok=True)
        
        # Multiprocess and consolidate intermediate metadata results
        print(f'\nStarting multiprocessing for freq={freq}, mode={mode}, '
              f'{dir_name}\n')
        chunks = 100
        nrows = md_df.shape[0]
        proc_args = [freq, mode, tmp_md_dir]
        
        executor = get_reusable_executor(max_workers=int(self.n_cpu*0.4), 
                                         timeout=120)
        with tqdm(total=nrows//chunks, colour='red') as pbar:
            for _,_ in enumerate(executor.map(self.generate_utterances, 
                    [proc_args + [
                        [row for _, row in md_df.iloc[i:i+chunks].iterrows()]] +
                            [i] for i in range(0, nrows, chunks)])):
                pbar.update()

        time.sleep(1)
        # Consolidate results from multiple processes saved inside 'tmp'
        print('\n\tConsolidating results...\n')
        # Save empty dataframes
        for md_name in md_dict:
            save_path = os.path.join(self.subset_md_path, f'{md_name}.csv')
            md_dict[md_name].to_csv(save_path, index=False)
        # Read tmp csvs
        tmp_csvs = glob.glob(os.path.join(tmp_md_dir, '*.csv'))
        for tc in tqdm(tmp_csvs, total=len(tmp_csvs), colour='green'):
            tmp_df = pd.read_csv(tc, header=None)
            if 'metrics' in tc:
                save_path = os.path.join(self.subset_md_path, 
                                         f'metrics_{dir_name}_{subdir}.csv')
            elif 'mixture' in tc:
                save_path = os.path.join(self.subset_md_path,
                                         f'mixture_{dir_name}_{subdir}.csv')
            tmp_df.to_csv(save_path, index=None, header=None, mode='a')
        os.system(f'rm -rf {tmp_md_dir}')

    
    def generate_utterances(self, all_args):
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
                # save_path = os.path.abspath(save_path)
                sf.write(save_path, src, freq)
                source_paths.append(save_path)
            return source_paths

        def loudness_normalize(sources, gains):
            return [source * gains[i] for i, source in enumerate(sources)]

        def resample_sources(sources_norm, freq):
            return [resample_poly(source, freq, self.RATE)
                    for source in sources_norm]

        def fit_lengths(sources_resampled, mode):
            if mode == 'min':
                target_len = min([len(source) for source in sources_resampled])
                final = []
                # Randomly trim longer source from start, equally or end
                trimway = random.randint(0, 2)
                for source in sources_resampled:
                    trimlen = len(source) - target_len
                    if not trimlen == 0:
                        if trimway == 0:
                            final.append(source[trimlen:])
                        elif trimway == 1:
                            final.append(
                                source[trimlen//2:-(trimlen-(trimlen//2))])
                        else:
                            final.append(source[:-trimlen])
                    else:
                        final.append(source)
                return final
            else:
                target_len = max([len(source) for source in sources_resampled])
                final = []
                # Randomly pad shorter source to the start, equally or end
                padway = random.randint(0, 2)
                for source in sources_resampled:
                    padlen = target_len - len(source)
                    if padway == 0:
                        final.append(np.pad(source, (padlen, 0)))
                    elif padway == 1:
                        final.append(
                            np.pad(source, (padlen//2, padlen-(padlen//2))))
                    else:
                        final.append(np.pad(source, (0, padlen)))
                return final
        
        def mix(sources_to_mix):
            mixture = np.zeros_like(sources_to_mix[0])
            for source in sources_to_mix:
                mixture += source
            return mixture

        def write_mix(mix_id, mixture, subdir, freq):
            filename = mix_id + '.wav'
            save_path = os.path.join(self.dir_path, subdir, filename)
            # save_path = os.path.abspath(save_path)
            sf.write(save_path, mixture, freq)
            return save_path

        def compute_snr_list(mixture, sources_to_mix):
            snr_list = []
            for i in range(len(sources_to_mix)):
                noise_min = mixture - sources_to_mix[i]
                snr_list.append(snr_xy(sources_to_mix[i], noise_min))
            return snr_list

        def snr_xy(x, y):
            return 10 * np.log10(
                np.mean(x ** 2) / (np.mean(y ** 2) + self.EPS) + self.EPS )
        
        """ Start of the func definition
        """
        freq = all_args[0]
        mode = all_args[1]
        tmp_md_dir = all_args[2]
        rows = all_args[3]
        it = all_args[4]

        res = []
        for row in rows:
            # Get sources and mixture infos
            mix_id, gains, sources = read_sources(row)
            # Transform sources
            transformed_sources = transform_sources(sources, freq, 
                                                    mode, gains)
            # Write sources and get their paths
            sources_path = write_sources(mix_id, transformed_sources, freq)

            for subdir in self.subdirs:
                if subdir == 'mix_clean':
                    sources_to_mix = transformed_sources[:self.n_src]
                elif subdir == 'mix_both':
                    sources_to_mix = transformed_sources
                elif subdir == 'mix_single':
                    sources_to_mix = [transformed_sources[0], 
                                    transformed_sources[-1]]
                else:
                    continue

                # Mix sources
                mixture = mix(sources_to_mix)
                # Write mixture and get its path
                mix_path = write_mix(mix_id, mixture, subdir, freq)
                length = len(mixture)
                # Compute SNR
                snr_list = compute_snr_list(mixture, sources_to_mix)
                res.append((mix_id, snr_list, mix_path, sources_path, 
                            length, subdir))
        
        self.save_tmp_results(res, it, tmp_md_dir)

    def save_tmp_results(self, res, it, tmp_md_dir):
        """ Save intermediate multiprocess results as csvs 
        """
        tdf_metrics = tdf_mixture = pd.DataFrame()
        for r in res:
            mix_id, snr_list, mix_path, sources_path, length, subdir = r
            tdf_metrics = tdf_metrics.append([[mix_id] + snr_list])
            if subdir == 'mix_clean':
                tdf_mixture = tdf_mixture.append(
                    [[mix_id, mix_path] + sources_path + [length]])
            # elif subdir == 'mix_single':
            #     tdf_mixture = tdf_mixture.append(
            #         [[mix_id, mix_path, sources_path[0]] + 
            #         noise_path + [length]])
            # else:
            #     tdf_mixture = tdf_mixture.append(
            #         [[mix_id, mix_path] + sources_path + 
            #         noise_path, + [length]])
        tdf_metrics.to_csv(os.path.join(
                tmp_md_dir, f'{os.getpid()}_{it}_metrics.csv'), 
            header=None, 
            index=None)
        tdf_mixture.to_csv(os.path.join(
                tmp_md_dir, f'{os.getpid()}_{it}_mixture.csv'), 
            header=None, 
            index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jvs_dir', type=str, required=True,
                        help='Absolute path of the JVS dataset directory')
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
    args = parser.parse_args()
    CreateJVSmixDataset(args)
