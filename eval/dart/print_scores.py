import os
import sys


if __name__ == '__main__':
    res = []
    metrics = []

    print(f'{"#" * 20} Summary {"#" * 20}')

    base_path = os.path.expanduser("~/temp_eval")

    path = os.path.join(base_path, 'bleu.txt')
    if os.path.isfile(path):
        with open(path) as f:
            bleu = float(f.read().strip().split()[2].replace(',', ''))
        print(f"BLEU: {bleu}")
        res.append(bleu)
        metrics.append('BLEU')

    path = os.path.join(base_path, 'meteor.txt')
    if os.path.isfile(path):
        with open(path) as f:
            meteor = float(f.readlines()[-1].strip().split()[-1])
        print(f"METEOR: {meteor}")
        res.append(meteor)
        metrics.append('METEOR')

    path = os.path.join(base_path, 'ter.txt')
    if os.path.isfile(path):
        with open(path) as f:
            ter = float(f.readlines()[-4].strip().split()[2])
        print(f"TER: {ter}")
        res.append(ter)
        metrics.append('TER')

    path = os.path.join(base_path, 'moverscore.txt')
    if os.path.isfile(path):
        with open(path) as f:
            moverscore = float(f.readlines()[-1].strip())
        print(f"MOVER: {moverscore}")
        res.append(moverscore)
        metrics.append('MOVER')

    path = os.path.join(base_path, 'bertscore.txt')
    if os.path.isfile(path):
        with open(path) as f:
            bertscore = float(f.read().strip().split()[-1])
        print(f"BERTScore: {bertscore}")
        res.append(bertscore)
        metrics.append('BERTScore')

        path = os.path.join(base_path, 'bleurt.txt')
    if os.path.isfile(path):
        with open(path) as f:
            scores = [float(s) for s in f.readlines()]
            bleurt = sum(scores) / len(scores)
        print(f"BLEURT: {bleurt}")
        res.append(bleurt)
        metrics.append('BLEURT')

    print('\n')
    print(' | '.join(metrics))
    print(' | '.join([f'{r:.2f}' for r in res]))

    if len(sys.argv) > 2:
        eval_file_path = os.path.splitext(sys.argv[1])[0] + '_eval.txt'
        with open(eval_file_path, 'w') as f:
            for m, v in zip(metrics, res):
                f.write(f'{m}: \t{v}')
                f.write('\n')
                f.write(' | '.join(metrics))
                f.write(' | '.join([f'{r:.2f}' for r in res]))
