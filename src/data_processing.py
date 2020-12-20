#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
import pandas as pd
from tqdm import tqdm
from string import digits
from pythainlp.tokenize import sent_tokenize
from pythainlp.tokenize import word_tokenize
from pythainlp.util import thai_digit_to_arabic_digit


def remove_url(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def remove_eng_alphabet(text):
    return text

def convert_thai_number(text):
    return thai_digit_to_arabic_digit(text)

def remove_number(text):
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)

def remove_emoji(text):
    return re.sub("\[.{0,12}\]","", text)

def remove_punctuation(text):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "", text)

def sent_tokenziation(text):
    return "\n".join(sent_tokenize(text, keep_whitespace=False))

def word_tokenization(text):
    return " ".join(word_tokenize(text, keep_whitespace=False))

def create_traindata(input_path, output_path):
    df = pd.read_csv(input_path, encoding = 'utf-8')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        title = " ".join(word_tokenize(row['title'], keep_whitespace=False)).replace(u'\xa0', u' ') + newline

        sum_sent = "\n".join(sent_tokenize(row['summary'], keep_whitespace=False))
        sum_word = " ".join(word_tokenize(sum_sent, keep_whitespace=False))

        body_sent = "\n".join(sent_tokenize(URLless_body, keep_whitespace=False))
        body_word = " ".join(word_tokenize(body_sent, keep_whitespace=False))

        # write these variables from each row (title, sum_word, body_word ) to a .txt file.
        file_object = open(output_path, 'a')
        file_object.write(title)
        file_object.write(sum_word)
        file_object.write(body_word)
        file_object.close()

if __name__ == '__main__':
    input_path = os.path.join(data/sample_data.csv)
    output_path = os.path.join(data/training_data.txt)

