1. log into midway machine:

username: user@midway2.rcc.uchicago.edu

password: My favoraite band's Guangzhou concert 

2. grab GPU nodes:

sinteractive -p gpu2 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --mem=8000 --time=1:00:00

3. load necessary modules:

module load cuda/9.0

conda activate tensorflow

4. if using jupyter notebook:

https://git.rcc.uchicago.edu/ivy2/Jupyter_on_compute_nodes
