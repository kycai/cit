# cit: concrete intelligent toolbox

The probabilistic machine learning models (e.g., [`ngboost`](https://github.com/stanfordmlgroup/ngboost)) were integrated with evolutionary multi-objective algorithms (e.g., NSGA-II, implemented in [`pymoo`](https://github.com/the-dual-space/pymoo)) to construct a two-step inverse design procedure for cement-based materials, involving binder-system design at the mortar level followed by concrete mixture-proportion optimization.

The developed models take not only the mixture proportion information but also material properties (serving as the "signature/fingerprint") of the binder materials, specimen geometry, and curing conditions as input features, with the output target being compressive strength.

Kangyi Cai 2026 @ Missouri S&T

## Environment setup

```bash
git clone https://github.com/kycai/cit.git
cd /path/to/cit

conda create --name cit python=3.12.12
conda activate cit

pip install -r requirements.txt
```

## Getting started

### Download checkpoints

The developed models and exemplary calculations can be downloaded from this [huggingface repository](https://huggingface.co/kycai23/cit_ngboost/tree/main). Please replace the `proj_scm_prob/data` folder in this code repository with the downloaded folder of the same name.

### Computation

The `prob_opt_search.py` implements the multi-/single-objective optimal search using the probabilistic ML models and evolutionary algorithms. The `prob_opt_analyze.ipynb` notebook is for post-processing and visualizing the optimization results.

## Citing

If you find **cit** useful in your research, we would appreciate it if you could cite our paper:

```markdown
[1] K. Cai, and H. Ma, "Intelligent design of multiphase supplementary cementitious material systems via probabilistic machine learning and evolutionary multi-objective algorithm," 2027.

[2] K. Cai, and H. Ma, "Machine learning-based intelligent proportion design of multi-phase supplementary cementitious materials," Journal of the Chinese Ceramic Society, vol. 54 no. 11, 2026. DOI: 10.14062/j.issn.0454-5648.20250880
```

Or use the following BibTeX entry:

```bibtex
@article{cai2027i,
  title={Intelligent design of multiphase supplementary cementitious material systems via probabilistic machine learning and evolutionary multi-objective algorithm},
  author={Cai, Kangyi and Ma, Hongyan},
  journal={},
  volume={},
  pages={},
  year={2027},
  doi={}
}

@article{cai2026m,
  title={Machine learning-based intelligent proportion design of multi-phase supplementary cementitious materials},
  author={Cai, Kangyi and Ma, Hongyan},
  journal={Journal of the Chinese Ceramic Society},
  volume={54},
  number={11},
  pages={},
  year={2026},
  doi={10.14062/j.issn.0454-5648.20250880}
}
```