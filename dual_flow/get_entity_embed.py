import sys
import torch
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def get_entity_embeds(args, dataloader, model):
    embeds = []
    for batch in tqdm(dataloader, desc="Get entity embedding"):
        with torch.no_grad():
            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.eval()
            outputs = model(**batch)
            embeds.append(outputs)
    embeds = torch.cat(embeds)
    return embeds

def save_embeds(embeds, idx_lst, output_dir):
    with open(output_dir, "wb") as outfile:
        pickle.dump((embeds.cpu().numpy(), idx_lst), outfile)
