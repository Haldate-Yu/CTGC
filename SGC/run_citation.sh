seed="12 22 32 42 52 62 72 82 92 102"

for i in $seed;
do
    # python citation.py --dataset cora --lr 0.2 --weight_decay 1e-5 --seed $i
    # python citation.py --dataset citeseer --lr 0.2 --weight_decay 1e-4 --seed $i
    python citation.py --dataset pubmed --lr 0.2 --weight_decay 2e-5 --seed $i
done

