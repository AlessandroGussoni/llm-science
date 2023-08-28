import numpy as np
from dataclasses import dataclass, field
import os


@dataclass
class CFG:
    train_path = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\train.csv"
    sciq_path = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\sciq.csv"
    full_sciq = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\full_sciq.csv"
    extra = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\extra_train_set.csv"
    extra6k = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\6000_train_examples.csv"
    extra_rlhf = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\llm_rlhf_extra.csv"
    stem1k = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\stem1k.csv"
    wiki = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\data\processed\wiki.csv"
    base_model = "microsoft/deberta-v3-large"
    base_model_mlm = r"C:\Users\Aless\OneDrive\Documenti\cuda-torch\deberta_mlm_ft"

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions

    def average_precision_at_k(actual, predicted, k):
        sorted_answer_indices = np.argsort(-predicted)
        top_answer_indices = sorted_answer_indices[:k]
        actual = int(actual)
        top_answer_indices = [int(i) for i in top_answer_indices]
        
        if actual in top_answer_indices:
            return [1, 0.5, 0.333333333333][top_answer_indices.index(actual)]
        else:
            return 0

    map_at_3_list = []
    for actual, predicted in zip(label_ids, predictions):
        ap_at_3 = average_precision_at_k(actual, predicted, k=3)
        map_at_3_list.append(ap_at_3)
    # Calculate the Mean Average Precision at 3 (MAP@3) using np.mean
    map_at_3 = np.mean(map_at_3_list)
    # Return a dictionary of metrics (including MAP@3)
    return {"MAP@3": map_at_3}

def getScore(trainer,df,token_ds):
    pred = trainer.predict(token_ds)
    
    map3_score = 0
    
    for index in range(df.shape[0]):
        columns = df.iloc[index].values
        scores = -pred.predictions[index]
        predict = np.array(list("ABCDE"))[np.argsort(scores)][:3].tolist()
        if columns[6] in predict:
            map3_score += [1,0.5,0.333333333333][predict.index(columns[6])]
    map3_score /= df.shape[0]
    print(f'score = {map3_score}')

    return map3_score

def wandb_setup(config):

    os.environ['WANDB_API_KEY'] = '0ce66896837e9c800adf4878b4dcde8c6c8a608f'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_NAME"] = config['wandb']['name']

    import wandb

    wandb.login()

    wandb.init(notes=config['wandb'].get("notes", ""),
               tags=config['wandb']['tags'])
