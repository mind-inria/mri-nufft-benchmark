# Benchmark of MRI-NUFFT 

This are a collection of script to perform benchmarking of MRI-NUFFT operations. 

They rely on the hydra configuration package and hydra-callback for measuring statistics. (see `requirements.txt`)


To fully reproduce the  benchmarks 4 steps are necessary: 

0. Get a Cartesian Reference image file, name `cpx_cartesian.npy`, you can use `python 0_create_data.py`
1. Generates the trajectory files  `python 00_trajectory.py` + shape of your data
2. Run the benchmarks. Currently are available: 
 - The Performance benchmark, checking the CPU/GPU usage and memory footprint for the different backend and configuration `perf` folder
    use `python 10_benchmark_perf.py`
    If you want to make several benchamrk in a row, you can run `python 50_auto_benchmark_perf.py` 
    Backends, trajectories and coils can be managed directly in the script (don't forget to install the necessary dependencies for each backend)
 - The Quality benchmark that check how the pair trajectory/backend performs for the reconstruction. in `qual` folder
3. Generate some analysis figures using `perf_analysis.py`




