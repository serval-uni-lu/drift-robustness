
for rq in lcld_rq1.1
do
  python -m drift_study.generate_config \
      -c config/${rq}_generate.yaml \
      -p output_path=config/${rq}.yaml
#   python -m drift_study.run_simulator -c config/logging.yaml -c config/${rq}.yaml
  python -m drift_study.visualization.line_graph -c config/${rq}.yaml -p output_file=./reports/lcld_201317_ds_time/${rq}.html -p plot_engine=plotly
done


for rq in lcld_rq1.2 lcld_rq1.3
do
  python -m drift_study.generate_config \
      -c config/${rq}_generate.yaml \
      -p output_path=config/${rq}.yaml
#   python -m drift_study.run_simulator -c config/logging.yaml -c config/${rq}.yaml
  python -m drift_study.visualization.table -c config/logging.yaml -c config/${rq}.yaml
done