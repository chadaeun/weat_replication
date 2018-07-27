import os
import json
import argparse


def main(args):
    weat_dict = dict()

    # build weat_dict
    for data_name in os.listdir(args.weat_dir):
        path = os.path.join(args.weat_dir, data_name)

        if os.path.abspath(path) == os.path.abspath(args.output):
            continue

        data_dict = dict()
        weat_dict[data_name] = data_dict
        keys = []

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    continue

                key, values = line.split(':')
                key = key.strip()
                values = [w.strip().lower() for w in values.split(',')]

                data_dict[key] = values
                keys.append(key)

        if len(keys) == 3:
            data_dict['method'] = 'wefat'

            data_dict['W_key'] = keys[0]
            data_dict['A_key'] = keys[1]
            data_dict['B_key'] = keys[2]

            data_dict['targets'] = '{}'.format(keys[0])
            data_dict['attributes'] = '{} vs {}'.format(keys[1], keys[2])

        elif len(keys) == 4:
            data_dict['method'] = 'weat'

            data_dict['X_key'] = keys[0]
            data_dict['Y_key'] = keys[1]
            data_dict['A_key'] = keys[2]
            data_dict['B_key'] = keys[3]

            data_dict['targets'] = '{} vs {}'.format(keys[0], keys[1])
            data_dict['attributes'] = '{} vs {}'.format(keys[2], keys[3])

    with open(args.output, 'w') as f:
        json.dump(weat_dict, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weat_dir', type=str, default='weat/', required=True,
                        help='WEAT data directory')
    parser.add_argument('--output', type=str, default='weat.json', required=True,
                        help='Output JSON file path')

    args = parser.parse_args()
    main(args)