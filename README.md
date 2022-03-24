# CALMS
Code and dataset for 'Contrastive Aligned Joint Learning for Multilingual Summarization'

## Organization
```
CALMS                        
├── fairseq                           # fairseq 0.9.0
├── myfairseq/examples/summarization    # fairseq user-dir
│   ├── criterions                      # loss functions    			
│   ├── data                            # dataset   
│   ├── models                          # nn model  
│   └── tasks                           # task definition
├── preprocess                        # process from raw data
│   ├── checkOverlap.py    	            # check overlap		
│   ├── dataClear.py                    # clear raw data                 	
│   └── dataSplit.py                    # split train/dev/test             
├── RELEASE-1.5.5                     # pyrouge dependencies
├── scripts                           # shell scripts for preprocess
│   ├── concat_lang5.sh                 # concat five languages
│   ├── filterdata.sh                   # filter raw data
│   └── predata.sh                      # tokenize dataset and create data-bin
├── utils                             # others
│   ├── trans                           # denoise preprocess
│   ├── calRouge.py                     # calculate multilingual rouge
│   ├── getbaseline.py                  # calculate lead and oracle
│   ├── logConfig.py                    # log class
│   ├── makeLabel.py                    # extract extractive label
│   └── tokenizer.py                    # tokenizer
├── finetune_joint.sh                 # jointly training multiple languages
├── finetune_lang5.sh                 # finetune a single language from the unified model
├── finetune.sh                       # finetune a single language from mBART
├── generate_zero.sh                  # generate from the unified model directly
├── rankSumm.sh                       # CSR: Contrastive Sentence Ranking
├── denoiseSumm.sh                    # SAS: Sentence Aligned Substitution
├── README.md                         
└── requirements.txt                  
```

## MLGSum Dataset

We provide three different versions for various requirements:

* [urls.tar.gz](https://drive.google.com/file/d/1i9xfOkQ60kixj0rZ-kCo8UCo2fZ51fCY/view?usp=sharing) is the archive urls of MLGSum and the basic script to crawl. However, some urls are lost due to the revision.
* [raw.tar.gz](https://drive.google.com/file/d/1SVdXbs59a1UBo-WyasjH0f9eI_kxZGYp/view?usp=sharing) is the original content from the web pages. You can run scripts/filterdata.sh to process it and get MLGSum.
* [clean.tar.gz](https://drive.google.com/file/d/1ZoOdEIDBGuG7ucdkjnGr5UVQRUkrE1pm/view?usp=sharing) is the final version of our MLGSum, which is reported in our paper. If you do not want to build the dataset by yourself, you can directly use this link.



## Dependencies

Our code is mainly based on Python3 and Fairseq 0.9. You should first run
```shell
$ pip install fairseq==0.9
$ pip install -r requirements.txt
```
to install the requirements. Besides, if you want to use [pyrouge](https://github.com/bheinzerling/pyrouge) for evaluation, you need to check some other dependencies:
```shell
$ sudo apt-get install libxml-perl libxml-dom-perl
$ pip install git+git://github.com/bheinzerling/pyrouge
```
The detailed information for ROUGE can be found in [MLROUGE](https://github.com/dqwang122/MLROUGE).

Note that the above steps have been included in training shell scripts (eg. *finetune.sh*). You can run them directly.

## Data Preprocess

You should tokenize and make bin files from MLGSum by yourself. We have provide the basic scripts (*scripts/predata.sh*) for you.

Pay attention, by default the mBART **does not** insert the language tags until the training phase. However, this makes it difficult for train multiple languages together. Thus, for the single language training, we follow the practice of mBART (*summarization_from_pretrained_mbart*), which does not insert the tags. And for the joint training, we insert the language tags during the data preprocess (using the **extended dictionary file**) and remove the insertion in training phrase (*summarization_from_pretrained_wo_langtag*).

Besides, for jointly training, the multiple languages should be mixed together. You can use the script (*scripts/concat_lang5.sh*) to concat and shuffle them. To conclude, your data-bin directory may look like this:
```
data-bin
├── wotag                # for the single language baseline
│   ├── de                     			
│   ├── en                            
│   ├── ru                          
│   ├── fr        
│   ├── ...                          
│   └── zh                          
├── wtag               # for jointly training with multiple languages
│   ├── lang5             # concat de+en+ru+fr+zh    			
│   ├── de                     			
│   ├── en                            
│   ├── ru                          
│   ├── fr        
│   ├── ...                          
└── └── zh
```

## Training

We build our model based on [Fairseq](https://github.com/pytorch/fairseq). Thus, to run the code, you should use fairseq repository and put the *myfairseq/examples/summarization* under the *fairseq/example* directory. You can also change the '--user-dir' path in 'finetine\*.sh' to avoid this operation.

The files in *myfairseq/examples/summarization/data* should be put under the *fairseq/fairseq/data/* and the *fairseq/fairseq/data/\_\_init\_\_.py* must be modified to include them. (We have done this for you.)

We define several new tasks for our training strategies:

* **summarization_from_pretrained_mbart**: finetune the monolingual summarization directly from mBART model.
* **summarization_from_pretrained_mbart_joint** jointly finetune multilingual summarization from mBART model.
* **summarization_from_pretrained_mbart_wo_tag** add the language tags during the data proprecess and ignore them during the training.
* **ranking_summaization** [Contrastive Sentence Ranking] create the (positive, negative) pairs and rank them.
* **denoise_summaization** [Sentence Aligned Substitution] use the translated lead sentences and insert them into the original document.
* **denoise_ranking_summaization** first replace translated sentences and then create ranking pairs.
* **ranking_denoise_summaization** first create ranking pairs and then replace translated sentences.

You can use different training strategies by changing the **--task** settings in the command line of Fairseq. We also provide several training scripts to use (**.sh*). The usage of each script can be found in each file.

## Generation

To generate summaries, you can use the **summarization_from_pretrained_mbart** for data without language tags and **summarization_from_pretrained_mbart_wo_tag** for data with language tags. You can also use **generate** mode in *finetune.sh* or *generate_zero.sh* for generation under different settings.


## Evaluation

We modify the original ROUGE for multilingual summarization. It is the same as [MLROUGE](https://github.com/dqwang122/MLROUGE). We put a simple version under *utils/calRouge.py* .
