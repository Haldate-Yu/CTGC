alpha="0.1 0.3 0.5 0.7"
aggr_type="ssgc ssgc_no_avg sgc"
norm_type="Arctan F2"

for a in $alpha;
do
    for agg in $aggr_type;
    do
        for norm in $norm_type;
        do
            python pyg_dataset.py --dataset Computers --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 2 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Computers --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 4 --alpha $a --aggr_type $agg --norm_type $norm            
            python pyg_dataset.py --dataset Computers --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 8 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Computers --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 16 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Computers --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 32 --alpha $a --aggr_type $agg --norm_type $norm
    
            python pyg_dataset.py --dataset Photo --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 2 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Photo --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 4 --alpha $a --aggr_type $agg --norm_type $norm            
            python pyg_dataset.py --dataset Photo --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 8 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Photo --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 16 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset Photo --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 32 --alpha $a --aggr_type $agg --norm_type $norm

            python pyg_dataset.py --dataset CS --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 2 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset CS --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 4 --alpha $a --aggr_type $agg --norm_type $norm            
            python pyg_dataset.py --dataset CS --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 8 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset CS --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 16 --alpha $a --aggr_type $agg --norm_type $norm
            python pyg_dataset.py --dataset CS --using_vg --vg 1 --lr 0.2 --weight_decay 5e-3 --method none --degree 32 --alpha $a --aggr_type $agg --norm_type $norm
       done
    done
done

