#!/bin/bash
# -*- coding: utf-8 -*-

for((i=1;i<280;i++))
do
python /home/tj816/mask-rcnn/train_data/json/json_to_dataset.py /home/tj816/mask-rcnn/train_data/json/${i}.json
done
