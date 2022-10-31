# Language Model Memorization vs Association

The code and data for "[Are Large Pre-Trained Language Models Leaking Your Personal Information?](https://arxiv.org/abs/2205.12628)" (Findings of EMNLP '22)

## Introduction

*Are Large Pre-Trained Language Models Leaking Your Personal Information?* We analyze whether Pre-Trained Language Models (PLMs) are prone to leaking personal information. Specifically, we query PLMs for email addresses with contexts of the email address or prompts containing the owner's name. We find that PLMs do leak personal information due to **memorization**. However, since the models are weak at **association**, the risk of specific personal information being extracted by attackers is low.

How does GPT-3 answer this question?
<img width="1559" alt="image" src="https://user-images.githubusercontent.com/47152740/198936706-cedccbb5-2b1c-415e-988c-7bffbb343686.png">

## Requirements

See `requirements.txt`

## Run

```
python pred.py
```

After this step, the models' predictions are stored as `.pkl` files in `results/`

To analyze the results in csv files and get the scores:

```
python analysis.py
```

*Note*: The scripts test the *0-shot setting* by default. Please edit the scripts, i.e., `settings =`, for evaluation on other settings.

## Data

Data are available at `data/`

`context.pkl` refers to the context setting

`{k}_shot_non_domain.pkl` refers to the setting when domain is unknown

`{k}_shot.pkl` refers to the setting when domain is known

`email2name.pkl` stores the mapping from email address to name

`name2email.pkl` stores the mapping from name to email address

`email_freq.pkl` stores the frequency of email address

## Citation

The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```
@inproceedings{huang2022large,
  title={Are Large Pre-Trained Language Models Leaking Your Personal Information?},
  author={Huang, Jie and Shao, Hanyin and Chang, Kevin Chen-Chuan},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
  year={2022}
}
```

