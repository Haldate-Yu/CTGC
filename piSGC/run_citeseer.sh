# settings
# lr=0.01/0.2/0.2
# wd=5e-4/5e-3/1e-4

lr=0.01
wd=5e-4

# parameters
alpha="0.1 0.3 0.5 0.7"
aggr_type="sgc ssgc ssgc_no_avg"
norm_type="Arctan F2 None"

for a in $alpha; do
  for ag in $aggr_type; do
    for n in $norm_type; do
      echo $ag $n $a
      python citation.py --dataset citeseer --using_vg --vg 1 --lr $lr --weight_decay $wd --method none --degree 2 --alpha $a --aggr_type $ag --norm_type $n
      python citation.py --dataset citeseer --using_vg --vg 1 --lr $lr --weight_decay $wd --method none --degree 4 --alpha $a --aggr_type $ag --norm_type $n
      python citation.py --dataset citeseer --using_vg --vg 1 --lr $lr --weight_decay $wd --method none --degree 8 --alpha $a --aggr_type $ag --norm_type $n
      python citation.py --dataset citeseer --using_vg --vg 1 --lr $lr --weight_decay $wd --method none --degree 16 --alpha $a --aggr_type $ag --norm_type $n
      python citation.py --dataset citeseer --using_vg --vg 1 --lr $lr --weight_decay $wd --method none --degree 32 --alpha $a --aggr_type $ag --norm_type $n
    done
  done
done
