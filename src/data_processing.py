#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
import pandas as pd
from tqdm import tqdm
from string import digits
from pythainlp.util import normalize
from pythainlp.tokenize import sent_tokenize
from pythainlp.tokenize import word_tokenize
from pythainlp.util import thai_digit_to_arabic_digit

def remove_url(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def remove_eng_alphabet(text):
    return re.sub("[a-z]+","", text)

def convert_thai_number(text):
    return thai_digit_to_arabic_digit(text)

def remove_number(text):
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)

def remove_emoji(text):
    return re.sub("\[.{0,12}\]","", text).replace(u'\xa0', u' ').replace(' )', '')

def remove_punctuation(text):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（#&$><]+", "", text)\

def sent_tokenziation(text):
    return "\n".join(sent_tokenize(text, keep_whitespace=False))

def word_tokenization(text):
    return " ".join(word_tokenize(normalize(text), keep_whitespace=False))

def pre_process(text):
    res = remove_url(text)
    res = remove_eng_alphabet(res)
    res = convert_thai_number(res)
    res = remove_number(res)
    res = remove_emoji(res)
    res = remove_punctuation(res)
    return res

def create_traindata(input_path, output_path):
    df = pd.read_csv(input_path, encoding = 'utf-8')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        title = pre_process(row['title'])
        title = word_tokenization(title) + "\n"

        summary = pre_process(row['summary'])
        summary = sent_tokenziation(summary)
        summary = word_tokenization(summary)

        body = pre_process(row['body'])
        body = sent_tokenziation(body)
        body = word_tokenization(body)

        # write these variables from each row (title, summary, body) to a .txt file.
        file_object = open(output_path, 'a')
        file_object.write(title)
        file_object.write(summary)
        file_object.write(body)
        file_object.close()

if __name__ == '__main__':
    input_path = os.path.join("/Users/Nakhun/kenlm/data/sample_data.csv")
    output_path = os.path.join("/Users/Nakhun/kenlm/data/training_data.txt")
    create_traindata(input_path, output_path)




