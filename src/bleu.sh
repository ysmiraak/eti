#!/usr/bin/env bash

# pip install --user sacrebleu
# sacrebleu --test-set wmt08/europarl --language-pair de-en --echo src > ../trial/data/test.txt

sacrebleu --input $1 ../trial/data/valid_tgt.txt
