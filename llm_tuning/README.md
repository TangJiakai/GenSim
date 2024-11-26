### LLM Tuning (SFT or PPO)
```
llm_tuning
├── code
│   ├── tune_llm.py
│   └── utils
│       ├── constants.py
│       └── utils.py
├── datasets
│   ├── ppo_data
│   │   └── ppo_data_template.json
│   └── sft_data
│       └── sft_data_template.json
├── README.md
└── scripts
    ├── cancel_tune.sh
    └── tune_llm.sh
```

- Save the `sft_data.json` and `ppo_data.json` under the `datasets` directory, in the `sft_data` and `ppo_data` folders respectively.
- Execute `tune_llm.sh` to start the tuning LLM, execute `cancel_tune.sh` to cancel the tuning process.
