import glob
from nltk import word_tokenize
import re, sys



def stardard_tokenize(sent):
    sent = ' '.join(re.split('(\W)', sent))
    sent = sent.split()
    sent = ' '.join(sent)
    return sent

# temp = glob.glob("./*beam")
# temp = glob.glob("../example/bart_webnlg.txt")
# for file in temp:
file = sys.argv[1]
print(file)
with open(file, 'r') as f:
    content = f.readlines()

print(content[-5:])
lower_case_tokenized_content = []
for sent in content:
    sent = stardard_tokenize(sent)
    lower_case_tokenized_content.append(sent)

print(lower_case_tokenized_content[-5:])

with open(file+'.lt2', 'w') as f:
    for line in lower_case_tokenized_content:
        print(line, file=f)


