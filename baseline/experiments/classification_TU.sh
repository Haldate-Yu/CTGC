
for i in MUTAG PTC_MR DD PROTEINS NCI1 NCI109  ENZYMES Mutagenicity FRANKENSTEIN IMDB-BINARY IMDB-MULTI COLLAB REDDIT-BINARY
do
	for j in gcn gin
	do
		python main.py  --cuda $1 --dataset $i  --model $j
	done
done



# for i in MUTAG PTC_MR DD PROTEINS NCI1 NCI109  ENZYMES Mutagenicity FRANKENSTEIN IMDB-BINARY IMDB-MULTI COLLAB REDDIT-BINARY

# do
# 	for j in gcn gin
# 	do
# 		python main.py  --cuda $1 --dataset $i --model $j
# 	done

# done
