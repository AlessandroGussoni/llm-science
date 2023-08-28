from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import gc
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, EarlyStoppingCallback
from sklearn.model_selection import KFold
from utils import CFG, compute_metrics, getScore, wandb_setup

from typing import Optional

from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType


def train(config):

    wandb_setup(config)

    df_train = pd.concat([
        pd.read_csv(CFG.extra6k),
        pd.read_csv(CFG.extra),
        pd.read_csv(CFG.sciq_path).loc[:1200],
        pd.read_csv(CFG.full_sciq).sample(frac=0.3, random_state=42),
        #pd.read_csv(CFG.stem1k),
        #pd.read_csv(CFG.wiki).sample(frac=0.25, random_state=42),
    ],
    axis=0)[['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']].sample(frac=1, random_state=42)

    df_train.reset_index(inplace=True, drop=True)
    df_train.dropna(inplace=True)
    print(df_train.shape)

    df_train.to_csv("data/training/df_train_v1.csv")

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)

    dataset = Dataset.from_pandas(df_train)

    valid_df = pd.read_csv(CFG.train_path)
    test_dataset = Dataset.from_pandas(valid_df)

    option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    index_to_option = {v: k for k,v in option_to_index.items()}

    def preprocess(example):
        first_sentence = [example['prompt']] * 5
        second_sentences = [example[option] for option in 'ABCDE']
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
        tokenized_example['label'] = option_to_index[example['answer']]
        
        return tokenized_example

    @dataclass
    class DataCollatorForMultipleChoice:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        
        def __call__(self, features):

            label_name = 'label' if 'label' in features[0].keys() else 'labels'
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]['input_ids'])
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
            ]


            flattened_features = sum(flattened_features, [])
            
            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors='pt',
            )
            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            batch['labels'] = torch.tensor(labels, dtype=torch.int64)
            return batch
        
    tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    training_args = TrainingArguments(
        **config['training_args']
)

    base_model = AutoModelForMultipleChoice.from_pretrained(CFG.base_model_mlm)

    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                             r=16,
                             lora_alpha=32,
                             lora_dropout=0.1)

    model = get_peft_model(base_model, peft_config)

    model.to("cuda")

    trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
)

    model.config.use_cache = False
    trainer.train()
    lmodel = trainer.model.merge_and_unload()
    
    lmodel.save_pretrained(config['output_dir'])

    getScore(trainer, 
            valid_df, 
            tokenized_test_dataset)

    del model
    del lmodel

    gc.collect()
    torch.cuda.empty_cache()

    test_pred = trainer.predict(tokenized_test_dataset).predictions
    predictions_as_ids = np.argsort(-test_pred, 1)
    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    predictions_as_string = valid_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]
    submission = valid_df[['prompt', 'answer', 'prediction']]
    submission.to_csv('submission2.csv', index=False)


if __name__ == '__main__':

    import json

    with open("training_scripts\config.json") as file:
        config = json.load(file)

    train(config)
