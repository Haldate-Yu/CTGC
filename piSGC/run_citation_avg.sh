seed="12 22 32 42 52 62 72 82 92 102"

for i in $seed;
do
    # python citation.py --dataset cora --using_vg --vg 1 --method none --lr 0.2 --weight_decay 1e-5 --seed $i --degree 16
    # python citation.py --dataset citeseer --using_vg --vg 1 --method none --lr 0.2 --weight_decay 1e-4 --seed $i --degree 8
    python citation.py --dataset pubmed --using_vg --vg 1 --method none --lr 0.2 --weight_decay 2e-5 --seed $i --degree 32
done

