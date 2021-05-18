import os
import re
import json
import argparse

def main(args):
    jsut_dir = args.jsut_dir
    mapping = {}
    # get a list of all directories in jsut dataset root directory
    subdirs = next(os.walk(jsut_dir))[1]
    subdirs.sort()
    
    for sd in subdirs:
        wavs = os.listdir(os.path.join(jsut_dir, sd, 'wav'))
        for w in wavs:
            w_ = re.sub('[^0-9]', '', w)
            mapping[w] = sd[:3]+w_
    
    with open('jsut_audio_naming.json', 'w') as fp:
        json.dump(mapping, fp)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsut_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)