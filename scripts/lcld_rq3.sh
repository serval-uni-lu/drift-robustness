delay=delays_none
for rq in rq3
do
  # OPTIMIZE
  # python \
  #   -m drift_study.optimize_simulator \
  #   -c config/logging.yaml \
  #   -c config/lcld_opt.yaml \
  #   -c config/rf_optimize.yaml \
  #   -c config/${delay}.yaml \
  #   -p sub_dir_path=opt_${delay}
  
  # FILTER CONFIG FOR EVAL
  python -m drift_study.complete_run \
    -p input_dir=data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/opt_${delay} \
    -p output_file=config/lcld_opt_${delay}.yaml \
    -p max_pareto=1
  
  # RUN EVAL
  # python \
  #   -m drift_study.run_simulator \
  #   -c config/logging.yaml \
  #   -c config/lcld_detector_eval.yaml \
  #   -c config/lcld_opt_${delay}.yaml \
  #   -c config/${delay}.yaml \
  #   -p sub_dir_path=${rq}

  # COPY REFERENCES
  cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq1.1/no_detection_rf_lcld_400.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/
  cp ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/rq1.3/periodic_42_*_rf_lcld_400_200000_0.json ./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq}/

  # # PLOT ON PARETO FRONT
  python -m drift_study.visualization.plot_ml_ntrain \
        -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/${rq} \
        -p output_file=reports/lcld_201317_ds_time/${rq}.html \
        -p max_n_train=143 \
        -p max_pareto_rank=1 \
        -p plot_engine=plotly
  
done