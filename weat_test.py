import argparse
import json
import os
import pandas as pd
from tabulate import tabulate

from lib import utils, weat


def main(args):
    # define get_word_vectors
    get_word_vectors = utils.define_get_word_vectors(args)

    # ready output file
    output_dir = os.path.split(os.path.abspath(args.output))[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    result_df = pd.DataFrame(columns=['Data Name', 'Targets', 'Attributes', 'Method', 'Score', '# of target words', '# of attribute words'])

    # compute WEAT score
    print('Computing WEAT score...')
    with open(args.weat_path) as f:
        weat_dict = json.load(f)

        for data_name, data_dict in weat_dict.items():
            if data_dict['method'] == 'wefat':
                print('{}: WEFAT is not implemented yet'.format(data_name))
                continue

                """
                W_key = data_dict['W_key']
                A_key = data_dict['A_key']
                B_key = data_dict['B_key']
                
                W = get_word_vectors(data_dict[W_key])
                A = get_word_vectors(data_dict[A_key])
                B = get_word_vectors(data_dict[B_key])

                A, B = utils.balance_word_vectors(A, B)
                
                num_target = len(W)
                num_attr = len(A)

                score = weat.wefat_score(W, A, B)
                p_value = weat.wefat_p_value(W, A, B)
                """

            elif data_dict['method'] == 'weat':
                X_key = data_dict['X_key']
                Y_key = data_dict['Y_key']
                A_key = data_dict['A_key']
                B_key = data_dict['B_key']

                X = get_word_vectors(data_dict[X_key])
                Y = get_word_vectors(data_dict[Y_key])
                A = get_word_vectors(data_dict[A_key])
                B = get_word_vectors(data_dict[B_key])

                X, Y = utils.balance_word_vectors(X, Y)
                A, B = utils.balance_word_vectors(A, B)

                num_target = len(X)
                num_attr = len(A)

                score = weat.weat_score(X, Y, A, B)
                # p_value = weat.weat_p_value(X, Y, A, B)

            else:
                print('{}: UNAVAILABLE METHOD \'{}\''.format(data_name, data_dict['method']))
                continue

            result_df = result_df.append(
                {
                    'Data Name': data_name,
                    'Targets': data_dict['targets'],
                    'Attributes': data_dict['attributes'],
                    'Method': data_dict['method'],
                    'Score': score,
                    '# of target words': num_target,
                    '# of attribute words': num_attr,
                },
                ignore_index=True
            )

    print('DONE')

    # print and write result
    print()
    print('Result:')
    print(tabulate(result_df, headers='keys', tablefmt='psql'))
    print()

    print('Writing result: {} ...'.format(args.output), end='')
    result_df.to_csv(args.output)
    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute WEAT score of pretrained word embedding models')
    parser.add_argument('--word_embedding_type', type=str, required=True,
                        help='Type of pretrained word embedding: word2vec, glove, tf-hub')
    parser.add_argument('--word_embedding_path', type=str, required=False,
                        help='Path of pretrained word embedding.')
    parser.add_argument('--weat_path', type=str, required=False, default='weat/weat.json',
                        help='Path of WEAT words file (weat.json)')
    parser.add_argument('--output', type=str, required=False, default='output/output.csv',
                        help='Path of output file (CSV formatted WEAT score)')
    parser.add_argument('--tf_hub', type=str, required=False,
                        help='Tensorflow Hub URL (ignored when word_embedding_type is not \'tf_hub\')')

    args = parser.parse_args()
    print('Arguments:')
    print('word_embedding_type:', args.word_embedding_type)
    print('word_embedding_path:', args.word_embedding_path)
    print('weat_path:', args.weat_path)
    print('output:', args.output)
    print('tf_hub:', args.tf_hub)
    print()

    main(args)