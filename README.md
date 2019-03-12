# SimIR
## Setup
The dependency packages:
  - Python 3
  - Tensorflow
  - Keras
  - gensim

Using anaconda to setup is an easy and good way.

## Experimental Evaluation
### GoogleCodeJam 
We descrive the use case of LSI.
When you use othe than LSI, please replace `LSI.py` with another file name.

1. change current directory to `GCJ/`
2. extract `gcj.tar.gz` and copy `gcj/` under `learning/`
3. change current directory to `learning/`
4. `python LSI.py`

We prepared `gcj.tar.gz` for this experiment, so we don't need the pre-processing phase.
If you try to generate the same as `gcj/`, you can get original source code of GoogelCodeJam from [DeepSim's repository](https://github.com/parasol-aser/deepsim) and use `GCJ/preprocess/prep.jar` for converting source codes to semantic representations.

### BigCloneBench

We descrive the use case of LSI.
When you use othe than LSI, please replace `lsi.py` with another file name, and change the argument of `learning.py`.

1. build BigCloneBench database according to https://github.com/clonebench/BigCloneBench
2. change current directory to `BCB/`
3. `python lsi.py`
4. open `lerning.py` and edit line 6-10 (write database connection settings)
5. `python learning.py lsi`

We prepared `functions.csv` for this experiment, so we don't need the pre-processing phase.

### Time Performance oF Generate Semantic Representation
Please change current directory to `preprocess/` and use `deepsim_encoder.jar` and `simir_encoder.jar`.
When you use dataset as same as the paper, please download from [MvnRepository](https://mvnrepository.com/)