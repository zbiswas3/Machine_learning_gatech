rem #!/bin/sh

rem # Replace 'X' below with the optimal values found
rem # If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

python run_experiment.py --pca --dataset1 --dim 5 --verbose --threads -1 > pca-dataset1-clustering.log 2>&1
python run_experiment.py --ica --dataset1 --dim 8 --verbose --threads -1 > ica-dataset1-clustering.log 2>&1
python run_experiment.py --rf  --dataset1 --dim 7 --verbose --threads -1 > rf-dataset1-clustering.log  2>&1
python run_experiment.py --rp  --dataset1 --dim 11 --verbose --threads -1 > rp-dataset1-clustering.log  2>&1

rem python run_experiment.py --pca --dataset2 --dim 6 --verbose --threads -1 > pca-dataset2-clustering.log 2>&1
rem python run_experiment.py --ica --dataset2 --dim 14 --verbose --threads -1 > ica-dataset2-clustering.log 2>&1
rem python run_experiment.py --rf  --dataset2 --dim 11 --verbose --threads -1 > rf-dataset2-clustering.log  2>&1
rem python run_experiment.py --rp  --dataset2 --dim 14 --verbose --threads -1 > rp-dataset2-clustering.log  2>&1

rem python run_experiment.py --pca --dataset1 --dim 5 --skiprerun --verbose --threads -1 > pca-dataset1-clustering.log 2>&1
rem python run_experiment.py --ica --dataset1 --dim 8 --skiprerun --verbose --threads -1 > ica-dataset1-clustering.log 2>&1
rem python run_experiment.py --rf  --dataset1 --dim 7 --skiprerun --verbose --threads -1 > rf-dataset1-clustering.log  2>&1
rem python run_experiment.py --rp  --dataset1 --dim 11 --skiprerun --verbose --threads -1 > rp-dataset1-clustering.log  2>&1

rem python run_experiment.py --pca --dataset2 --dim 6 --skiprerun --verbose --threads -1 > pca-dataset2-clustering.log 2>&1
rem python run_experiment.py --ica --dataset2 --dim 14 --skiprerun --verbose --threads -1 > ica-dataset2-clustering.log 2>&1
rem python run_experiment.py --rf  --dataset2 --dim 11 --skiprerun --verbose --threads -1 > rf-dataset2-clustering.log  2>&1
rem python run_experiment.py --rp  --dataset2 --dim 14 --skiprerun --verbose --threads -1 > rp-dataset2-clustering.log  2>&1

rem python run_experiment.py --pca --dataset1 --dim 18  --verbose --threads -1 > pca-dataset1-clustering.log 2>&1
rem python run_experiment.py --ica --dataset1 --dim 48  --verbose --threads -1 > ica-dataset1-clustering.log 2>&1
rem python run_experiment.py --rf  --dataset1 --dim 10  --verbose --threads -1 > rf-dataset1-clustering.log  2>&1
rem python run_experiment.py --rp  --dataset1 --dim 18  --verbose --threads -1 > rp-dataset1-clustering.log  2>&1

rem python run_experiment.py --pca --dataset2 --dim 5  --verbose --threads -1 > pca-dataset2-clustering.log 2>&1
rem python run_experiment.py --ica --dataset2 --dim 11  --verbose --threads -1 > ica-dataset2-clustering.log 2>&1
rem python run_experiment.py --rf  --dataset2 --dim 7  --verbose --threads -1 > rf-dataset2-clustering.log  2>&1
rem python run_experiment.py --rp  --dataset2 --dim 8  --verbose --threads -1 > rp-dataset2-clustering.log  2>&1

