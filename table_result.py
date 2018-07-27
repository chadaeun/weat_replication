import pandas as pd
import argparse
import os
import json


def main(args):
    file_names = os.listdir(args.output_dir)
    result_df = pd.DataFrame(columns=['Data Name', 'Targets', 'Attributes'] + file_names, index=None)

    # key: (target, attr)
    # value: row number in result_df
    data_idx = {}

    # set targets and attributes
    with open(args.weat_path) as f:
        weat_dict = json.load(f)

        for data_name, data_dict in weat_dict.items():
            # WEFAT score is not implemented yet
            if data_dict['method'] != 'weat':
                continue

            data_idx[data_name] = len(result_df)

            result_df = result_df.append(
                pd.Series([data_name, data_dict['targets'], data_dict['attributes']], index=['Data Name', 'Targets', 'Attributes']),
                ignore_index=True
            )

    # write result
    for file_idx, file_name in enumerate(file_names, 3):
        file_path = os.path.join(args.output_dir, file_name)
        output_df = pd.read_csv(file_path)

        for i, row in output_df.iterrows():
            result_idx = data_idx[row['Data Name']]
            result_df.iloc[result_idx, file_idx] = row['Score']

    # save image
    result_df.to_csv(args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='output/',
                        help='Directory of weat_score.py output files')
    parser.add_argument('--weat_path', type=str, required=False, default='weat/weat.json',
                        help='WEAT json file path (weat.json)')
    parser.add_argument('--result_path', type=str, required=False, default='result.csv',
                        help='Result CSV file path')
    args = parser.parse_args()

    main(args)