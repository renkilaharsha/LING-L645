#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 17:21:21 2022

@author: praneeth44
"""

from collections import defaultdict
import math

corpus = """are you still here ?
where are you ?
are you tired ?
i am tired .
are you in england ?
were you in mexico ?"""

sentences = corpus.split('\n')

unigram = defaultdict(int)

for i in sentences:
    for j in i.split():
        unigram[j] += 1
        unigram['<VAL>'] += 1


def t1():
    return defaultdict(int)


bigram = defaultdict(t1)

for i in sentences:
    k = ['<BOS>'] + i.split()
    for j in range(1, len(k)):
        bigram[k[j - 1]][k[j]] += 1
        bigram[k[j - 1]]['<VAL>'] += 1

print(bigram)

for keys, values in bigram.items():
    for key, val in values.items():
        if key != '<VAL>':
            bigram[keys][key] = (bigram[keys][key] + 1) / (bigram[keys]['<VAL>'] + len(unigram) - 1)

test = """where are you ?
were you in england ?
are you in mexico ?
i am in mexico .
are you still in mexico ?"""


def perp_bi(sents):
    model_p = 1
    t_len = 0
    tok = set()
    for i in sents:
        k = ['<BOS>'] + i.split()
        for j in range(1, len(k)):
            t_len += 1
            tok.add(k[j])
            if k[j] in bigram[k[j - 1]]:
                model_p = model_p * bigram[k[j - 1]][k[j]]
            else:
                model_p = model_p / (bigram[k[j - 1]]['<VAL>'] + len(unigram) - 1)
    return model_p ** (-1 / t_len)


def perp_uni(sents):
    model_p = 1
    t_len = 0
    tok = set()
    for i in sents:
        for j in range(len(k)):
            t_len += 1
            tok.add(k[j])
            if unigram[k[j]] == 0:
                model_p = model_p / (unigram['<VAL>'])
            else:
                model_p = model_p * unigram[k[j]] / unigram['<VAL>']
    return model_p ** (-1 / t_len)


sentences = test.split('\n')

ans = []

model_p = 1
t_len = 0
for i in sentences:
    val = 1
    k = ['<BOS>'] + i.split()
    for j in range(1, len(k)):
        t_len += 1
        if k[j] in bigram[k[j - 1]]:
            val = val * bigram[k[j - 1]][k[j]]
            model_p = model_p * bigram[k[j - 1]][k[j]]
        else:
            val = val / (bigram[k[j - 1]]['<VAL>'] + len(unigram))
            model_p = model_p / (bigram[k[j - 1]]['<VAL>'] + len(unigram))
    ans.append([i, val, math.log(val), val ** (-1 / len(unigram))])

ans.sort(key=lambda x: -x[1])
# for i in ans:
#     print(i)
# print(model_p**(-1/t_len))

print(bigram)

print('bigram test', perp_bi(test.split('\n')))
print('bigram train', perp_bi(corpus.split('\n')))

print('unigram test', perp_uni(test.split('\n')))
print('unigram train', perp_uni(corpus.split('\n')))


