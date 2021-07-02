import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from transformers import ViTModel


class ViT_NumScore(nn.Module):
    def __init__(self):
        super(ViT_NumScore, self).__init__()
        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.bert = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        self.conv_images = nn.Conv2d(768, 512, 1)
        self.conv_text = nn.Conv2d(768, 512, 1)
        self.scoring_i2t = nn.Linear(197, 1)
        self.scoring_t2i = nn.Linear(197, 1)

    def forward(self, images_inputs, input_ids, token_type_ids, attention_masks, val=False):
        images_embeddings = self.image_model(images_inputs).last_hidden_state
        del images_inputs
        text_embeddings = self.bert({'input_ids': input_ids,
                                    'token_type_ids': token_type_ids,
                                    'attention_mask': attention_masks})['sentence_embedding']
        del input_ids, token_type_ids
        images_embeddings = images_embeddings.unsqueeze(3).permute(0, 2, 1, 3)
        images_embeddings = self.conv_images(images_embeddings).squeeze()
        images_embeddings = images_embeddings.permute(0, 2, 1)
        text_embeddings = text_embeddings.unsqueeze(2).unsqueeze(3)
        text_embeddings = self.conv_text(text_embeddings).squeeze()
        if val:
            return images_embeddings, text_embeddings
        # similarity
        # images_embeddings_norm = F.normalize(images_embeddings, dim=-1)
        # text_embeddings_norm = F.normalize(text_embeddings, dim=-1)
        images_embeddings_norm = images_embeddings/images_embeddings.norm(dim=2)[:, :, None]
        text_embeddings_norm = text_embeddings/text_embeddings.norm(dim=1)[:, None]
        i2t = torch.matmul(images_embeddings, text_embeddings_norm.transpose(0, 1))
        t2i = torch.matmul(images_embeddings_norm, text_embeddings.transpose(0, 1))
        i2t, t2i = i2t.transpose(1, 2), t2i.transpose(1, 2)
        i2t = self.scoring_i2t(i2t).squeeze()
        t2i = self.scoring_t2i(t2i).squeeze().transpose(0, 1)

        return i2t, t2i
