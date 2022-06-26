if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

# gcn gin baselines ...
#for num in {0..4}; do
#  echo $num
#  for i in MUTAG; do
#    for j in gcn; do
#      python main.py --cuda $1 --dataset $i --model $j
#    done
#  done
#
#  # sgc ssgc
#  for i in MUTAG; do
#    for j in sgc ssgc; do
#      for k in 3 6 9; do
#        python main.py --cuda $1 --dataset $i --model $j --K $k
#      done
#    done
#  done

#  for i in MUTAG; do
#    for j in gin_w; do
#      # python main.py --cuda $1 --dataset $i --model $j
#      python main.py --cuda $1 --dataset $i --model $j --pinv True
#    done
#  done
#
#done

#for i in MUTAG; do
#  for j in gin; do
#    python main.py --cuda $1 --dataset $i --model $j
#  done
#done
# 666 for adj filter
# MUTAG PTC_MR PROTEINS
for i in MUTAG PTC_MR PROTEINS; do
  for j in gin_w; do
    for k in 10 20 30; do
      # python main.py --cuda $cuda --dataset $i --model $j
      python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
    done
  done
done

#for i in MUTAG; do
#  for j in gcn; do
#    python main.py --cuda $1 --dataset $i --model $j
#  done
#done

# sgc ssgc
#for i in MUTAG; do
#  for j in sgc ssgc; do
#    for k in 3 6 9; do
#      python main.py --cuda $1 --dataset $i --model $j --K $k
#    done
#  done
#done

# ctgc
#for i in MUTAG PROTEINS NCI1; do
#for i in MUTAG; do
#  for j in ssgc ssgc_no_avg sgc; do
#    for k in arctan f2 none; do
#      for l in 3 6 9; do
#        for a in 0.1 0.3 0.5 0.7; do
#          python main.py --cuda $1 --dataset $i --model ctgc --K $l --aggr_type $j --norm_type $k --alpha $a --pinv True
#        done
#      done
#    done
#  done
#done

#for i in MUTAG PTC_MR DD PROTEINS NCI1 NCI109 ENZYMES Mutagenicity FRANKENSTEIN IMDB-BINARY IMDB-MULTI COLLAB REDDIT-BINARY; do
#  for j in gcn gin; do
#    python main.py --cuda $1 --dataset $i --model $j
#  done
#
#done
