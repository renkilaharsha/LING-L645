import sys, math, re, pickle
from collections import defaultdict, Counter


def tokenise(s):
    # !!! Implement a tokenisation function !!!
    s = re.sub('([^a-zA-Z0-9\']+)', ' \g<1> ', s.strip()).split()
    return s


model = defaultdict(lambda: defaultdict(float))

bigrams, unigrams = defaultdict(Counter), Counter()  # Unigram and bigram counts
k = 1

def calculate_probability(S, unigrams, bigrams,model):

    if(S[0] + " " + S[1] in bigrams):
        numerator = bigrams[S[0] + " " + S[1]]
    else:
        numerator = 1

    if S[0] in unigrams:
        denominator = unigrams[S[0]]
    else:
        denominator = 1
    if(S[0] in model):
        model[S[0]][S[1]] = float(numerator / denominator)
    else:
        model[S[0]] = {}
        model[S[0]][S[1]] = float(numerator / denominator)
   # print(S[0],S[1])
    #print(numerator,denominator, bigrams[S[0]+ " " + S[1]])
    return float(numerator / denominator)


def n_gram_dict(output, tokens, n):
    for i in range(len(tokens) - n + 1):
        g = ' '.join(tokens[i:i + n])
        print(g)
        output.setdefault(g, 0)
        output[g] += 1
    return output


line = sys.stdin.readline()
semtence  = []
while line:  # Collect counts from standard input
    semtence.append(line)
    tokens = ['<BOS>'] + tokenise(line)
    unigrams.update(tokens)
    n_gram_dict(bigrams, tokens, 2)
    # !!! Collect bigram and unigram counts !!!

    line = sys.stdin.readline()

print(bigrams)



def calculate_sentence_probability(S):
    tokens = ['<BOS>'] + tokenise(S)

    final_probabaility = 1
    for i in range(len(tokens) - 1):
        prob = calculate_probability(tokens[i:i + 2], unigrams, bigrams,model)
        # print(tokens[i:i+2],prob)
        final_probabaility *= prob

    print('%.6f\t%.6f\t' % (final_probabaility, math.log(final_probabaility)), ['<BOS>'] + tokenise(S))
    return final_probabaility

def calculate_sentence_perplexity(S):
    tokens = ['<BOS>'] + tokenise(S)
    final_probabaility = 1
    for i in range(len(tokens) - 1):
        prob = calculate_probability(tokens[i:i + 2], unigrams, bigrams, model)
        # print(tokens[i:i+2],prob)
        final_probabaility *= prob

    perplex_prob = 1/final_probabaility
    perplex_prob = perplex_prob**(-1/len(tokens))
    print('%.6f\t%.6f\t' % (perplex_prob, math.log(perplex_prob)), ['<BOS>'] + tokenise(S))

#print(model)
sentences = ["where are you?", "were you in england?", "are you in mexico?", "i am in mexico.",
             "are you still in mexico?"]

for sent in sentences:
    calculate_sentence_probability(sent)

print("---------------------------------------")
print(" Probability of sentence using perplexity")

for sent in sentences:
    calculate_sentence_perplexity(sent)
    # !!! Now calculate the probabilities !!!
#print(model.items())
print('Saved %d bigrams.' % sum([len(i) for i in model.items()]))
pickle.dump(dict(model), open('model_ngram.lm', 'wb'))
