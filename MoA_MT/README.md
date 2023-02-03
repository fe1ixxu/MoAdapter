This is the repo for our intern project: [Hierarchical Mixture of Experts (HMoE)](https://fb.quip.com/afXgAiMJrYek)

## Building VirtualEnvironments:
```
conda create -n fairseq-moe python=3.8
pip install -e ./
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout origin/experts_lt_gpus_moe_reload_fix
pip install .
```
