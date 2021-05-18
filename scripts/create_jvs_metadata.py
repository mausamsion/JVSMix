import os
import argparse
import soundfile as sf
import pandas as pd
import glob
import json
from tqdm import tqdm

RATE = 24000    # all audio in jvs dataset is 24KHz

def main(args):
    jvs_dir = args.jvs_dir
    md_dir = args.md_dir
    create_jvs_metadata(jvs_dir, md_dir)

def create_jvs_metadata(jvs_dir, md_dir):
    speakers = pd.read_csv(os.path.join(jvs_dir, 'gender_f0range.txt'), 
                           delimiter=' ')
    speakers = speakers[speakers.columns[:2]]
    speakers = speakers.set_index('speaker').to_dict()['Male_or_Female']
    
    # jvs_metadata = {}
    # for speaker in speakers.keys():
    #     sound_files = glob.glob(os.path.join(jvs_dir, speaker, '**/*.wav'),
    #                             recursive=True)
    #     parallel_sound_files = []
    #     nonparallel_sound_files = []
    #     for sf in sound_files:
    #         if 'parallel' in sf:
    #             parallel_sound_files.append(sf.replace(jvs_dir+'/', ''))
    #         elif 'nonpara' in sf:
    #             nonparallel_sound_files.append(sf.replace(jvs_dir+'/', ''))
    #     jvs_metadata[speaker] = {'gender': speakers[speaker], 
    #                              'parallel_data': parallel_sound_files, 
    #                              'nonparallel_data': nonparallel_sound_files}
    
    # with open('../metadata/jvs_metadata.json', 'w') as fp:
    #     for ch in json.dumps(jvs_metadata, indent=4):
    #         fp.write(ch)

    jvs_metadata = pd.DataFrame()
    for speaker in speakers.keys():
        sound_files = glob.glob(os.path.join(jvs_dir, speaker, '**/*.wav'),
                                recursive=True)
        for sound in sound_files:
            dur = round(len(sf.SoundFile(sound)) / RATE, 2)
            if 'parallel' in sound:
                row = [speaker, speakers[speaker], 'parallel',
                       dur, sound.replace(jvs_dir+'/', '')]
            elif 'nonpara' in sound:
                row = [speaker, speakers[speaker], 'non-parallel',
                       dur, sound.replace(jvs_dir+'/', '')]
            jvs_metadata = jvs_metadata.append([row])
    jvs_metadata.columns = ['speaker_id', 'gender', 'subset', 
                            'duration', 'path']
    jvs_metadata.to_csv(os.path.join(md_dir, 'jvs_metadata.csv'), index=None)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jvs_dir', type=str, required=True, 
                        help='Absolute path of the JVS dataset directory')
    parser.add_argument('--md_dir', type=str, required=True, 
                        help='Absolute path to metadata directory')
    args = parser.parse_args()
    main(args)
