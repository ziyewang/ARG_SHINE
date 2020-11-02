# ARG-SHINE: improve antibiotic resistance class prediction by integrating Sequence Homology, functIoNal information, and deep convolutional nEural network

## <a name="started"></a>Installation

### <a name="docker"></a>Conda

We recommend using conda to run ARG-SHINE.

### <a name="docker"></a>Obtain ARG-SHINE and create an environment
After installing Anaconda (or miniconda), fisrt obtain ARG-SHINE:

```sh
git clone https://github.com/ziyewang/ARG_SHINE
```
Then simply create a ARGshine environment 

```sh
cd ARG-SHINE
conda env create -f argshine.yaml
source activate ARGshine
```
You can run ARG_SHINE against a fasta file (protein sequences) with the following commmand
```sh
python ARG_SHINE_predict.py test.fasta
```

