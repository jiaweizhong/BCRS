#!/bin/bash
python3 evaluate_tiny.py --res ./pred.json --gt ../../../../../TinyPerson/mini_annotations/tiny_set_test_all.json --detail --metric 'ap'
cat ./scores.txt
