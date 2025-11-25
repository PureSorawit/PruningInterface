[Averaging] - basically sum of importance score

python averaging.py --prune 0.30


[Intersecting] - conjugation of mask
```
python intersecting.py --target 0.30
```


[applying mask] - zeroing out weight
```
python apply_mask.py --mask output/combined_masks.json --save vit_b16_cifar100_average_pruned.pth
python apply_mask.py --mask output/final_intersect_mask.json --save vit_b16_cifar100_intersect_pruned.pth
```


### Eval - the accuracy is a bit weird right now
[Baseline]
```
python eval.py
```


[pruned]
```
python eval.py --ckpt vit_b16_cifar100_average_pruned.pth
python eval.py --ckpt vit_b16_cifar100_intersect_pruned.pth
```