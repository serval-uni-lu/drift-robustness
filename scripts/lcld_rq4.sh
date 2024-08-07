# YOU NEED TO RUN THE OPTIMIZATION AND FILTER OF RQ2 BEFORE EXECUTING RQ4

# PERIODIC RETRAINING FOR BASELINE
rq=lcld_rq4.0
# python -m drift_study.generate_config \
#       -c config/${rq}_generate.yaml \
#       -p output_path=config/${rq}.yaml
# python -m drift_study.run_simulator -c config/logging.yaml -c config/${rq}.yaml


delay=delays_half
rq=rq4.1

# OPTIMIZE
# USES RQ2 OPTIMIZATION

# FILTER CONFIG FOR EVAL
# USES RQ2 FILTER

# RUN EVAL
# python \
#   -m drift_study.run_simulator \
#   -c config/logging.yaml \
#   -c config/lcld_detector_eval.yaml \
#   -c config/lcld_opt_delays_all.yaml \
#   -c config/${delay}.yaml \
#   -p sub_dir_path=${rq}

# COPY REFERENCES
cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq1.1/no_detection_rf_lcld_400.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/
cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq4.0/periodic_42_*_rf_lcld_400_200000_2W.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/

# # PLOT ON PARETO FRONT
python -m drift_study.visualization.plot_ml_ntrain \
        -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq} \
        -p output_file=reports/lcld_201317_ds_time/${rq}.html \
        -p max_n_train=143 \
        -p max_pareto_rank=1 \
        -p plot_engine=plotly


delay=delays_twice
rq=rq4.2

# RUN EVAL
# python \
#   -m drift_study.run_simulator \
#   -c config/logging.yaml \
#   -c config/lcld_detector_eval.yaml \
#   -c config/lcld_opt_delays_all.yaml \
#   -c config/${delay}.yaml \
#   -p sub_dir_path=${rq}

# COPY REFERENCES
cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq1.1/no_detection_rf_lcld_400.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/
cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq4.0/periodic_42_*_rf_lcld_400_200000_8W.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/

# # PLOT ON PARETO FRONT
python -m drift_study.visualization.plot_ml_ntrain \
        -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq} \
        -p output_file=reports/lcld_201317_ds_time/${rq}.html \
        -p max_n_train=143 \
        -p max_pareto_rank=1 \
        -p plot_engine=plotly