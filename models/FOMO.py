# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------
# Modified from Transformers: 
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py
# ------------------------------------------------------------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTConfig, OwlViTModel
from transformers.models.owlvit.modeling_owlvit import *

from .utils import *
from .few_shot_dataset import FewShotDataset, aug_pipeline, collate_fn

from util import box_ops
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json


def split_into_chunks(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


class UnkDetHead(nn.Module):
    def __init__(self, method, known_dims, att_W, **kwargs):
        super(UnkDetHead, self).__init__()
        print("UnkDetHead", method)
        self.method = method
        self.known_dims = known_dims
        self.att_W = att_W
        self.process_mcm = nn.Softmax(dim=-1)

        if "sigmoid" in method:
            self.process_logits = nn.Sigmoid()
            self.proc_obj = True
        elif "softmax" in method:
            self.process_logits = nn.Softmax(dim=-1)
            self.proc_obj = True
        else:
            self.proc_obj = False

    def forward(self, att_logits):
        logits = att_logits @ self.att_W
        k_logits = logits[..., :self.known_dims]
        unk_logits = logits[..., self.known_dims:].max(dim=-1, keepdim=True)[0]
        logits = torch.cat([k_logits, unk_logits], dim=-1)
        objectness = torch.ones_like(unk_logits).squeeze(-1)

        if "mean" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.mean(dim=-1, keepdim=True)[0]

        elif "max" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.max(dim=-1, keepdim=True)[0]

        if "mcm" in self.method:
            mcm = self.process_mcm(k_logits).max(dim=-1, keepdim=True)[0]
            objectness *= (1 - mcm)

        if self.proc_obj:
            objectness -= objectness.mean()
            objectness /= objectness.std()
            objectness = torch.sigmoid(objectness)

        return logits, objectness.squeeze(-1)


class OwlViTTextTransformer(OwlViTTextTransformer):
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        num_samples, seq_len = input_shape  # num_samples = batch_size * num_max_text_queries
        # OWLVIT's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(num_samples, seq_len).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # take features from the end of tokens embedding (end of token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(torch.tensor(float("-inf")))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OurOwlViTModel(OwlViTModel):
    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward_vision(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get embeddings for all text queries in all batch samples

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return image_embeds, vision_outputs

    def forward_text(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            """

        # Get embeddings for all text queries in all batch samples
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_embeds_norm, text_outputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_base_image_embeds: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:
            """
        # Use OWL-ViT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # normalized features
        image_embeds, vision_outputs = self.forward_vision(pixel_values=pixel_values,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        text_embeds_norm, text_outputs = self.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        # cosine similarity as logits and set it on the correct device
        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if return_base_image_embeds:
            warnings.warn(
                "`return_base_image_embeds` is deprecated and will be removed in v4.27 of Transformers, one can"
                " obtain the base (unprojected) image embeddings from outputs.vision_model_output.",
                FutureWarning,
            )
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
        else:
            text_embeds = text_embeds_norm

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FOMO(nn.Module):
    """This is the OWL-ViT model that performs open-vocabulary object detection"""
    def __init__(self, args, model_name, known_class_names, unknown_class_names, templates, image_conditioned, device):
        """ Initializes the model.
        Parameters:
            model_name: the name of the huggingface model to use
            known_class_names: list of the known class names
            templates:
            attributes: dict of class names (keys) and the corresponding attributes (values).

        """
        super().__init__()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.owlvit = OurOwlViTModel.from_pretrained(model_name).to(device)
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)

        self.known_class_names = known_class_names
        self.unknown_class_names = unknown_class_names
        all_classnames = known_class_names + unknown_class_names
        self.all_classnames = all_classnames
        self.templates = templates
        self.num_attributes_per_class = args.num_att_per_class

        if image_conditioned:
            fs_dataset = FewShotDataset(
                args.dataset,
                args.image_conditioned_file,
                self.known_class_names,
                args.num_few_shot,
                self.processor,
                args.data_task,
                aug_pipeline,
            )

            fs_dataloader = DataLoader(dataset=fs_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=True)

            if args.use_attributes:
                with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.attributes_file}', 'r') as f:
                    attributes = json.loads(f.read())

                self.attributes_texts = [f"object which (is/has/etc) {cat} is {a}" for cat, att in attributes.items() for a in att]
                self.att_W = torch.rand(len(self.attributes_texts), len(known_class_names), device=device)

                with torch.no_grad():
                    mean_known_query_embeds, embeds_dataset = self.get_mean_embeddings(fs_dataset)
                    text_mean_norm, att_query_mask = self.prompt_template_ensembling(self.attributes_texts, templates)
                    self.att_embeds = text_mean_norm.detach().clone().to(device)
                    self.att_query_mask = att_query_mask.to(device)

                if args.att_selection:
                    self.attribute_selection(embeds_dataset, args.neg_sup_ep * 500, args.neg_sup_lr)
                    selected_idx = torch.where(torch.sum(self.att_W, dim=1) != 0)[0]
                    self.att_embeds = torch.index_select(self.att_embeds, 1, selected_idx)
                    self.att_W = torch.index_select(self.att_W, 0, selected_idx)
                    print(f"Selected {len(selected_idx.tolist())} attributes from {len(self.attributes_texts)}")
                    self.attributes_texts = [self.attributes_texts[i] for i in selected_idx.tolist()]

                self.att_W = F.normalize(self.att_W, p=1, dim=0).to(device)
                self.att_query_mask = None

                if args.att_adapt:
                    self.adapt_att_embeddings(mean_known_query_embeds)

                if args.att_refinement:
                    self.attribute_refinement(fs_dataloader, args.neg_sup_ep, args.neg_sup_lr)

                if args.use_attributes:
                    self.att_embeds = torch.cat([self.att_embeds, torch.matmul(self.att_embeds.squeeze().T, self.att_W).mean(1, keepdim=True).T.unsqueeze(0)], dim=1)
                else:
                    with torch.no_grad():
                        unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([self.att_embeds, unknown_query_embeds], dim=1)

                eye_unknown = torch.eye(1, device=self.device)
                self.att_W = torch.block_diag(self.att_W, eye_unknown)
            else:
                ## run simple baseline
                with torch.no_grad():
                    mean_known_query_embeds, _ = self.get_mean_embeddings(fs_dataset)
                    unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([mean_known_query_embeds, unknown_query_embeds], dim=1)

                self.att_W = torch.eye(len(known_class_names) + 1, device=device)
                self.att_query_mask = None
        else:
            self.att_embeds, self.att_query_mask = self.prompt_template_ensembling(all_classnames, templates)
            self.att_W = torch.eye(len(all_classnames), device=self.device)

        self.unk_head = UnkDetHead(args.unk_method, known_dims=len(known_class_names),
                                   att_W=self.att_W, device=device)

    def attribute_refinement(self, fs_dataloader, epochs, lr):
        optimizer = torch.optim.AdamW([self.att_embeds], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.att_embeds.requires_grad_()
        # Create tqdm object for displaying progress
        pbar = tqdm(range(epochs), desc="Refining selected attributes:")
        for _ in pbar:
            mean_loss = []
            for batch_idx, batch in enumerate(fs_dataloader):
                optimizer.zero_grad()
                with torch.no_grad():
                    image_embeds, targets = self.image_guided_forward(batch["image"].to(self.device),
                                                                      bboxes=batch["bbox"],
                                                                      cls=batch["label"])
                    if image_embeds is None:
                        continue
                    targets = torch.stack(targets).to(self.device)

                cos_sim = cosine_similarity(image_embeds, self.att_embeds, dim=-1)
                logits = torch.matmul(cos_sim, self.att_W)
                loss = criterion(logits, targets)  # Compute loss
                loss.backward()
                optimizer.step()  # Update cls_embeds using gradients
                mean_loss.append(loss.detach().cpu().numpy())

            # Update progress bar with current mean loss
            pbar.set_postfix({"loss": np.mean(mean_loss)}, refresh=True)
        self.att_embeds.requires_grad_(False)
        return

    def attribute_selection(self, fs_dataloader, epochs, lr):
        target_embeddings = []
        class_ids = []
        for class_id, embeddings_batches in fs_dataloader.items():
            for batch in embeddings_batches:
                target_embeddings.append(batch)
                class_ids.extend([class_id] * batch.shape[0])

        # Concatenate target embeddings
        image_embeddings = torch.cat(target_embeddings, dim=0).to(self.device)

        # Create one-hot encoded targets
        num_classes = len(fs_dataloader)
        targets = F.one_hot(torch.tensor(class_ids), num_classes=num_classes).float().to(self.device)

        optimizer = torch.optim.AdamW([self.att_W], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.att_W.requires_grad_()
        lambda1 = 0.01

        # Create tqdm object for displaying progress
        pbar = tqdm(range(epochs), desc="Attribute selection:")
        for _ in pbar:
            optimizer.zero_grad()
            self.att_W.data = torch.clamp(self.att_W.data, 0, 1)
            cos_sim = cosine_similarity(image_embeddings, self.att_embeds, dim=-1)
            logits = torch.matmul(cos_sim, self.att_W)

            loss = criterion(logits, targets)  # Compute loss
            l1_reg = torch.norm(self.att_W, p=1)
            loss += lambda1 * l1_reg
            loss.backward()
            optimizer.step()  # Update cls_embeds using gradients
            pbar.set_postfix({"loss": loss}, refresh=True)

        with torch.no_grad():
            _, top_indices = torch.topk(self.att_W.view(-1), num_classes * self.num_attributes_per_class)
            self.att_W.fill_(0)  # Reset all attributes to 0
            self.att_W.view(-1)[top_indices] = 1

        self.att_W.requires_grad_(False)
        return

    def get_mean_embeddings(self, fs_dataset):
        dataset = {i: [] for i in range(len(self.known_class_names))}
        for img_batch in split_into_chunks(range(len(fs_dataset)), 3):
            image_batch = collate_fn([fs_dataset.get_no_aug(i) for i in img_batch])
            grouped_data = defaultdict(list)

            for bbox, label, image in zip(image_batch['bbox'], image_batch['label'], image_batch['image']):
                grouped_data[label].append({'bbox': bbox, 'image': image})

            for l, data in grouped_data.items():
                tmp = self.image_guided_forward(torch.stack([d["image"] for d in data]).to(self.device),
                                                [d["bbox"] for d in data]).cpu()
                dataset[l].append(tmp)

        return torch.cat([torch.cat(dataset[i], 0).mean(0) for i in range(len(self.known_class_names))], 0).unsqueeze(
            0).to(self.device), dataset

    def adapt_att_embeddings(self, mean_known_query_embeds):
        self.att_embeds.requires_grad_()  # Enable gradient computation
        optimizer = torch.optim.AdamW([self.att_embeds], lr=1e-3)  # Define optimizer
        criterion = torch.nn.MSELoss()  # Define loss function

        for i in range(1000):
            optimizer.zero_grad()  # Clear gradients

            output = torch.matmul(self.att_W.T.unsqueeze(0), self.att_embeds)
            loss = criterion(output, mean_known_query_embeds)  # Compute loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update cls_embeds using gradients

            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss.item()}")

        self.att_embeds.requires_grad_(False)

    def prompt_template_ensembling(self, classnames, templates):
        print('performing prompt ensembling')
        text_sum = torch.zeros((1, len(classnames), self.model.owlvit.text_embed_dim)).to(self.device)

        for template in templates:
            print('Adding template:', template)
            # Generate text for each class using the template
            class_texts = [template.replace('{c}', classname.replace('_', ' ')) for classname in
                           classnames]

            text_tokens = self.processor(text=class_texts, return_tensors="pt", padding=True, truncation=True).to(
                self.device)

            # Forward pass through the text encoder
            text_tensor, query_mask = self.forward_text(**text_tokens)

            text_sum += text_tensor

        # Calculate mean of text embeddings
        # text_mean = text_sum / text_count
        text_norm = text_sum / torch.linalg.norm(text_sum, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_norm, query_mask

    def embed_image_query(
            self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor,
            each_query_boxes
    ) -> torch.FloatTensor:
        _, class_embeds = self.model.class_predictor(query_image_features)
        pred_boxes = self.model.box_predictor(query_image_features, query_feature_map)
        pred_boxes_as_corners = box_ops.box_cxcywh_to_xyxy(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device
        bad_indexes = []
        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor(each_query_boxes[i], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)
            else:
                bad_indexes.append(i)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None
        return query_embeds, box_indices, pred_boxes, bad_indexes

    def image_guided_forward(
            self,
            query_pixel_values: Optional[torch.FloatTensor] = None, bboxes=None, cls=None
    ):
        # Compute feature maps for the input and query images
        # save_tensor_as_image_with_bbox(query_pixel_values[0].cpu(), bboxes[0][0], f'tmp/viz/{cls}_img.png')
        query_feature_map = self.model.image_embedder(pixel_values=query_pixel_values)[0]
        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        # Get top class embedding and best box index for each query image in batch
        query_embeds, _, _, missing_indexes = self.embed_image_query(query_image_feats, query_feature_map, bboxes)
        if query_embeds is None:
            return None, None
        query_embeds /= torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        if cls is not None:
            return query_embeds, [item for index, item in enumerate(cls) if index not in missing_indexes]

        return query_embeds

    def forward_text(
            self,
            input_ids,
            attention_mask,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None, ):

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        text_embeds, text_outputs = self.model.owlvit.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   return_dict=return_dict)

        text_embeds = text_embeds.unsqueeze(0)

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.unsqueeze(0)
        query_mask = input_ids[..., 0] > 0

        return text_embeds.to(self.device), query_mask.to(self.device)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> OwlViTObjectDetectionOutput:

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        # Embed images and text queries
        _, vision_outputs = self.model.owlvit.forward_vision(pixel_values=pixel_values,
                                                             output_attentions=output_attentions,
                                                             output_hidden_states=output_hidden_states,
                                                             return_dict=return_dict)

        # Get image embeddings
        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )

        image_embeds = image_embeds.reshape(new_size)

        batch_size, num_patches, num_patches, hidden_dim = image_embeds.shape
        image_feats = torch.reshape(image_embeds, (batch_size, num_patches * num_patches, hidden_dim))

        # Predict object boxes
        pred_boxes = self.model.box_predictor(image_feats, image_embeds)

        (pred_logits, class_embeds) = self.model.class_predictor(image_feats, self.att_embeds.repeat(batch_size, 1, 1),
                                                     self.att_query_mask)

        out = OwlViTObjectDetectionOutput(
            image_embeds=image_embeds,
            text_embeds=self.att_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            vision_model_output=vision_outputs,
        )

        out.att_logits = out.logits  #TODO: remove later
        out.logits, out.obj = self.unk_head(out.logits)
        return out


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, model_name, pred_per_im=100, image_resize=768, device='cpu', method='regular'):
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.pred_per_im = pred_per_im
        self.method=method
        self.image_resize = image_resize
        self.device = device
        self.clip_boxes = lambda x, y: torch.cat(
            [x[:, 0].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 1].clamp_(min=0, max=y[0]).unsqueeze(1),
             x[:, 2].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 3].clamp_(min=0, max=y[0]).unsqueeze(1)], dim=1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, viz=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if viz:
            reshape_sizes = torch.Tensor([[self.image_resize, self.image_resize]]).repeat(len(target_sizes), 1)
            target_sizes = (target_sizes * self.image_resize / target_sizes.max(1, keepdim=True).values).long()
        else:
            max_values, _ = torch.max(target_sizes, dim=1)
            reshape_sizes = max_values.unsqueeze(1).repeat(1, 2)

        if self.method =="regular":
            results = self.post_process_object_detection(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "attributes":
            results = self.post_process_object_detection_att(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "seperated":
            results = self.post_process_object_detection_seperated(outputs=outputs, target_sizes=reshape_sizes)

        for i in range(len(results)):
            results[i]['boxes'] = self.clip_boxes(results[i]['boxes'], target_sizes[i])
        return results

    def post_process_object_detection(self, outputs, target_sizes=None):
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, logits, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, boxes)
        return results

    def post_process_object_detection_att(self, outputs, target_sizes=None):
        ## this post processing should produce the same predictions as `post_process_object_detection`
        ## but also report what are the most dominant attribute per class (used to produce some of the
        ## figures in the MS
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob_att = torch.sigmoid(outputs.att_logits)
        
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
        
        def get_known_objs(prob, logits, prob_att, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]

            # Get the batch indices and prediction indices to index into prob_att
            batch_indices = torch.arange(logits.shape[0]).view(-1, 1).expand_as(topk_indexes)
            pred_indices = topk_boxes

            # Gather the attributes corresponding to the top-k labels
            # You will gather along the prediction dimension (dim=1)
            gathered_attributes = prob_att[batch_indices, pred_indices, :]

            # Now gather the boxes in a similar way as before
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # Combine the results into a list of dictionaries
            return [{'scores': s, 'labels': l, 'boxes': b, 'attributes': a} for s, l, b, a in zip(scores, labels, boxes, gathered_attributes)]
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, prob_att, boxes)
        return results

    def post_process_object_detection_seperated(self, outputs, target_sizes=None):
        ## predicts the known and unknown objects seperately. Used when the known and unknown classes are
        ## derived one from text and the other from images.

        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj.squeeze(-1)

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, out_logits, boxes):
            # import ipdb; ipdb.set_trace()
            scores, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im//2, dim=1)
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        def get_unknown_objs(obj, out_logits, boxes):

            scores, topk_indexes = torch.topk(obj.unsqueeze(-1), self.pred_per_im//2, dim=1)
            scores = scores.squeeze(-1)
            labels = torch.ones(scores.shape, device=scores.device) * out_logits.shape[-1]
            # import ipdb; ipdb.set_trace()
            boxes = torch.gather(boxes, 1, topk_indexes.repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob[..., :-1].clone(), logits[..., :-1].clone(), boxes)
        unknown_results = get_unknown_objs(prob[..., -1].clone(), logits[..., :-1].clone(), boxes)

        out = []
        for k, u in zip(results, unknown_results):
            out.append({
                "scores": torch.cat([k["scores"], u["scores"]]),
                "labels": torch.cat([k["labels"], u["labels"]]),
                "boxes": torch.cat([k["boxes"], u["boxes"]])
            })
        return out

def build(args):
    device = torch.device(args.device)

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.classnames_file}', 'r') as file:
        ALL_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.prev_classnames_file}', 'r') as file:
        PREV_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    CUR_KNOWN_ClASSNAMES = [cls for cls in ALL_KNOWN_CLASS_NAMES if cls not in PREV_KNOWN_CLASS_NAMES]

    known_class_names = PREV_KNOWN_CLASS_NAMES + CUR_KNOWN_ClASSNAMES

    if args.unk_proposal and args.unknown_classnames_file != "None":
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.unknown_classnames_file}', 'r') as file:
            unknown_class_names = sorted(file.read().splitlines())
        unknown_class_names = [k for k in unknown_class_names if k not in known_class_names]
        unknown_class_names = [c.replace('_', ' ') for c in unknown_class_names]

    else:
        unknown_class_names = ["object"]

    if args.templates_file:
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.templates_file}', 'r') as file:
            templates = file.read().splitlines()
    else:
        templates = ["a photo of a {c}"]

    model = FOMO(args, args.model_name, known_class_names, unknown_class_names,
                 templates, args.image_conditioned, device)

    postprocessors = PostProcess(args.model_name, args.pred_per_im, args.image_resize, device, method=args.post_process_method)
    return model, postprocessors
