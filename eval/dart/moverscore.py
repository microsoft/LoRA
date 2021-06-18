from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
import sys
import statistics

if __name__ =='__main__':
    if len(sys.argv)<3:
        print('usage: python moverscore.py [references.txt] [hypothesis.txt]')
        exit(1)
    references = [r.strip('\n') for r in  open(sys.argv[1]).readlines()]
    translations =[t.strip('\n') for t in  open(sys.argv[2]).readlines()]
    print(f'ref len:{len(references)}')
    print(f'hyp len:{len(translations)}')
    idf_dict_hyp = get_idf_dict(translations) # idf_dict_hyp = defaultdict(lambda: 1.)#translations is a list of candidate sentences
    idf_dict_ref = get_idf_dict(references) # idf_dict_ref = defaultdict(lambda: 1.)
    #reference is a list of reference sentences
    scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
    print(statistics.mean(scores))
