#!/bin/bash

start=20
end=25
for ((i=$start;i<=$end;i++))
do
	#f="python3 test.py -s ${i} --seq2seq_weight ./weight/seq2seq_weight_e1350_p2000 --folder ../trajectories_dataset/valid/ --fps 120"
	#echo ${f}
	#eval ${f}
	f="python3 test.py -s ${i} --seq2seq_weight ./weight/seq2seq_weight_e70_p50000 --folder ../trajectories_dataset/valid/ --fps 120"
	echo ${f}
	eval ${f}
done
