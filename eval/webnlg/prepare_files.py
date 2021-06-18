import os
import sys

def prepare_files_ter(system_output, reference_file0, reference_file1, reference_file2):
    """
    Generate files for METEOR and TER input.
    :param inputdir: directory with bleu files
    :return:
    """
    references = []  # each element is a list of references
    pure_references = []
    # initialref = inputdir + 'test-webnlg-all-notdelex.lex'
    # complete refs with references for all sents
    with open(reference_file0, 'r') as f:
        for i, line in enumerate(f):
            references.append([line.strip() + ' (id' + str(i) + ')\n'])
            pure_references.append([line])

    # create a file with only one reference for TER
    # with open('all-notdelex-oneref-ter.txt', 'w+') as f:
    #     for ref in references:
    #         f.write(''.join(ref))

    # TODO: This is for multiple references?
    # files = [(inputdir, filename) for filename in os.listdir(inputdir)]
    # for filepath in files:
    #     if 'all-notdelex-reference' in filepath[1] and 'reference0' not in filepath[1]:
    #         with open(filepath[0]+filepath[1], 'r') as f:
    #             for i, line in enumerate(f):
    #                 if line != '\n':
    #                     references[i].append(line.strip() + ' (id' + str(i) + ')\n')
    #                     pure_references[i].append(line)
    for filepath in [reference_file1, reference_file2]:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line != '\n':
                    references[i].append(line.strip() + ' (id' + str(i) + ')\n')
                    pure_references[i].append(line)

    with open('all-notdelex-refs-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    # prepare generated hypotheses
    with open(system_output, 'r') as f:
        geners = [line.strip() + ' (id' + str(i) + ')\n' for i, line in enumerate(f)]
    with open('relexicalised_predictions-ter.txt', 'w+') as f:
        f.write(''.join(geners))

    # data for meteor
    # For N references, it is assumed that the reference file will be N times the length of the test file,
    # containing sets of N references in order.
    # For example, if N=4, reference lines 1-4 will correspond to test line 1, 5-8 to line 2, etc.
    with open('all-notdelex-refs-meteor.txt', 'w+') as f:
        for ref in pure_references:
            empty_lines = 8 - len(ref)  # calculate how many empty lines to add (8 max references)
            f.write(''.join(ref))
            if empty_lines > 0:
                f.write('\n' * empty_lines)
    # print('Input files for METEOR and TER generated successfully.')


if __name__ == "__main__":
    system_output = sys.argv[1]
    reference_file0 = sys.argv[2]
    reference_file1 = sys.argv[3]
    reference_file2 = sys.argv[4]
    # print(sys.argv)
    prepare_files_ter(system_output, reference_file0, reference_file1, reference_file2)
