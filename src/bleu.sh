#!/usr/bin/env bash

# pip install --user sacrebleu
# sacrebleu -t wmt08/europarl -l de-en --echo src > ../trial/data/test.txt

sacrebleu -b -i $1 ../trial/data/valid_tgt.txt
