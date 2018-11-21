#!/usr/bin/env bash

# ~/.local/bin/sacrebleu --test-set wmt08/europarl --language-pair de-en --echo src > ../trial/data/test.txt

~/.local/bin/sacrebleu --input $1 ../trial/data/valid_tgt.txt
