# ARG-SHINE: improve antibiotic resistance class prediction by integrating Sequence Homology, functIoNal information, and deep convolutional nEural network

## <a name="requirement"></a>Requirements
### <a name="InterProScan"></a>InterProScan
Users need to install InterProScan before using ARG-SHINE. The documents for the InterProScan installation are available at https://github.com/ebi-pf-team/interproscan/wiki/HowToDownload.
### <a name="Git LFS"></a>[Git LFS](https://git-lfs.github.com/)

## <a name="started"></a>Installation

### <a name="docker"></a>Conda

We recommend using conda to run ARG-SHINE.

### <a name="docker"></a>Obtain ARG-SHINE and create an environment
After installing Anaconda (or miniconda), fisrt obtain ARG-SHINE:

```sh
git lfs clone https://github.com/ziyewang/ARG_SHINE
```
Then simply create an argshine_env environment 

```sh
cd ARG-SHINE
conda env create -f argshine_env.yaml
conda activate argshine_env
```
You can run ARG_SHINE against a fasta file (protein sequences) with the following commmand
```sh
bash argshine_prediction.sh fasta_file output_path PATH_to_interproscan
example: bash argshine_prediction.sh test_data/test.fasta test_data $HOME/interproscan-5.44-79.0
```

## <a name="preprocessing"></a>Contacts and bug reports
Please send bug reports or questions to Ziye Wang: zwang17@fudan.edu.cn 

## <a name="preprocessing"></a>References
[1] Arango-Argoty, G., Garner, E., Pruden, A., Heath, L.S., Vikesland, P., Zhang, L.: Deeparg: a deep learning
approach for predicting antibiotic resistance genes from metagenomic data. Microbiome 6(1), 1–15 (2018)

[2] Hamid, M.N., Friedberg, I.: Transfer learning improves antibiotic resistance class prediction. BioRxiv (2020)
