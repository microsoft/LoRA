import statistics
import sys

if __name__ == '__main__':
    print('##################### Summary ##########################')

    with open('bleu_seen.txt') as f:
        bleu_seen = float(f.read().strip().split()[2].replace(',',''))
    print("BLEU Seen: {:.2f}".format(bleu_seen))

    with open('bleu_unseen.txt') as f:
        bleu_unseen = float(f.read().strip().split()[2].replace(',',''))
    print("BLEU Unseen: {:.2f}".format(bleu_unseen))

    with open('bleu_all.txt') as f:
        bleu_all = float(f.read().strip().split()[2].replace(',',''))
    print("BLEU All: {:.2f}".format(bleu_all))

    with open('meteor_seen.txt') as f:
        meteor_seen = float(f.readlines()[-1].strip().split()[-1])
    print("METEOR Seen: {:.2f}".format(meteor_seen))

    with open('meteor_unseen.txt') as f:
        meteor_unseen = float(f.readlines()[-1].strip().split()[-1])
    print("METEOR Unseen: {:.2f}".format(meteor_unseen))

    with open('meteor_all.txt') as f:
        meteor_all = float(f.readlines()[-1].strip().split()[-1])
    print("METEOR All: {:.2f}".format(meteor_all))

    with open('ter_seen.txt') as f:
        ter_seen = float(f.readlines()[-4].strip().split()[2])
    print("TER Seen: {:.2f}".format(ter_seen))

    with open('ter_unseen.txt') as f:
        ter_unseen = float(f.readlines()[-4].strip().split()[2])
    print("TER Unseen: {:.2f}".format(ter_unseen))

    with open('ter_all.txt') as f:
        ter_all = float(f.readlines()[-4].strip().split()[2])
    print("TER All: {:.2f}".format(ter_all))


    print(' & '.join(["{:.2f}".format(bleu_seen), "{:.2f}".format(bleu_unseen), "{:.2f}".format(bleu_all),
                      "{:.2f}".format(meteor_seen), "{:.2f}".format(meteor_unseen), "{:.2f}".format(meteor_all),
                      "{:.2f}".format(ter_seen), "{:.2f}".format(ter_unseen), "{:.2f}".format(ter_all)]))





