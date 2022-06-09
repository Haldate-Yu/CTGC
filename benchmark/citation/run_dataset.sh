lr=0.01
wd=1e-3

#echo "PubMed"
#echo "====="

# echo "sgc"
# python sgc.py --dataset=Computers --K=2 --lr=$lr --weight_decay=$wd

# echo "ssgc"
#python ssgc.py --dataset=Computers --K=2 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=4 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=8 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=16 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=32 --lr=$lr --weight_decay=$wd

#echo "ctgc"
#alpha="0.1 0.3 0.5 0.7"
#norm="arctan f2 none"
#aggr="ssgc ssgc_no_avg sgc"
#
#for a in $alpha; do
#  for n in $norm; do
#    for ag in $aggr; do
#      echo $ag $n $a
#      python ctgc.py --dataset=PubMed --K=2 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#      python ctgc.py --dataset=PubMed --K=4 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#      python ctgc.py --dataset=PubMed --K=8 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#      python ctgc.py --dataset=PubMed --K=16 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#      python ctgc.py --dataset=PubMed --K=32 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#    done
#  done
#done

echo "Computers"
echo "====="

# echo "sgc"
# python sgc.py --dataset=Computers --K=2 --lr=$lr --weight_decay=$wd

# echo "ssgc"
#python ssgc.py --dataset=Computers --K=2 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=4 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=8 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=16 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Computers --K=32 --lr=$lr --weight_decay=$wd

echo "ctgc"
alpha="0.1 0.3 0.5 0.7"
norm="arctan f2 none"
aggr="ssgc ssgc_no_avg sgc"

for a in $alpha; do
  for n in $norm; do
    for ag in $aggr; do
      echo $ag $n $a
      python ctgc.py --dataset=Computers --K=2 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Computers --K=4 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Computers --K=8 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Computers --K=16 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Computers --K=32 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
    done
  done
done
#
echo "Photo"
echo "====="

#echo "sgc"
#python sgc.py --dataset=Photo --K=2 --lr=$lr --weight_decay=$wd

#echo "ssgc"
#python ssgc.py --dataset=Photo --K=2 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Photo --K=4 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Photo --K=8 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Photo --K=16 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=Photo --K=32 --lr=$lr --weight_decay=$wd

for a in $alpha; do
  for n in $norm; do
    for ag in $aggr; do
      echo $ag $n $a
      python ctgc.py --dataset=Photo --K=2 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Photo --K=4 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Photo --K=8 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Photo --K=16 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
      python ctgc.py --dataset=Photo --K=32 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
    done
  done
done

#echo "CS"
#echo "====="
#
#echo "sgc"
#python sgc.py --dataset=CS --K=2 --lr=$lr --weight_decay=$wd
#
#echo "ssgc"
#python ssgc.py --dataset=CS --K=2 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=CS --K=4 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=CS --K=8 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=CS --K=16 --lr=$lr --weight_decay=$wd
#python ssgc.py --dataset=CS --K=32 --lr=$lr --weight_decay=$wd

#for a in $alpha
#do
#    for n in $norm
#    do
#        for ag in $aggr
#        do
#            echo $ag $n $a
#            python ctgc.py --dataset=CS --K=2 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#            python ctgc.py --dataset=CS --K=4 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#            python ctgc.py --dataset=CS --K=8 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#            python ctgc.py --dataset=CS --K=16 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#            python ctgc.py --dataset=CS --K=32 --lr=$lr --weight_decay=$wd --alpha=$a --aggr_type=$ag --norm_type=$n
#        done
#    done
#done
