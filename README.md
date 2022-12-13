# ZARTS: On Zero-order Optimization for Neural Architecture Search
Implementation of [ZARTS](https://arxiv.org/abs/2110.04743). 

# Quick Start
## How to search an architecture
You can search on CIFAR-10 by running the following codes:

`bash scripts/run_search.sh`

If you want to search on other spaces, you can change the value of `SPACE` in `scripts/run_search.sh` and run the above code.

## How to train the discovered architecture
After searching, the code will save the genotype of discovered architectures in the directory `ckpt`, where $ID is the timestamp of your search process. Then you can evluate the discovered architecture by training the discovered architecture for 600 epochs from scratch. Run the following code.

`bash scritps/run_fulltrain.sh`

Notice that before running the above code, you should change `$ID` in `scripts/run_fulltrain.sh` as the timestamp of your search process.

# Citations
<pre><code>@article{wang2021zarts,
  title={ZARTS: On Zero-order Optimization for Neural Architecture Search},
  author={Wang, Xiaoxing and 
	  Guo, Wenxuan and 
	  Su, Jianlin and 
	  Yang, Xiaokang and
          Yan, Junchi},
  journal={arXiv preprint arXiv:2110.04743},
  year={2021}
}</code></pre>
