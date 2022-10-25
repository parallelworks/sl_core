#=====================================
# SuperLearner launch script
#=====================================
# Run the SuperLearner

python -m sl_main.py \
       --conda_sh '~/.miniconda3/etc/profile.d/conda.sh' \
       --superlearner_conf '../sample_inputs/whondrml_global_train_25_inputs_update.csv' \
       --n_jobs '8' \
       --num_inputs '25' \
       --cross_val_score 'True' \
       --model_dir '../sample_outputs/model_dir' \
       --hpo 'True' \
       --data '../sample_inputs/whondrml_global_train_25_inputs_update.csv' \
       --backend 'loky'
