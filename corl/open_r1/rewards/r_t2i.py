import re
import torch
import torchmetrics.functional as metric_F
from torchvision import transforms
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pycocoevalcap.spice.spice import Spice
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.cider.cider import Cider
# import language_evaluation
# coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
from transformers import AutoConfig
from transformers import BlipProcessor, BlipForConditionalGeneration
from corl.open_r1.rewards.bert_score.bert_score_wrapper import BertScoreWrapper, BertSimCSEWrapper
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .r_utils import (
    safe_string_equal,
    extract_answer_text_from_qa,
    extract_answer_letter_from_response,
    extract_answer_text_from_response,
    token_level_max_match_similarity,
    soft_jaccard,
)


class T2ICycleConsistencyReward:
    def __init__(self, task_args):
        self.args = task_args

        self.cap_cs_metrics = task_args.caption_cs_metrics
        self.using_simcse = task_args.using_simcse
        self.img_cs_metrics = task_args.image_cs_metrics
        self.using_img_cs = task_args.using_image_cs
        self.using_external_caption_model = task_args.using_external_caption_model

        if self.using_img_cs:
            if "lpips" in self.img_cs_metrics:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor()
                ])

        self.blip_model = None
        self.blip_processor = None
        self.bert_scorer = None
        self.lpips_metric = None

    @torch.inference_mode()
    def generate_caption_with_policy_mmgpt(
            self, images, cap_mmgpt=None, processing_class=None,
    ):
        device = cap_mmgpt.device

        # generate captions
        # task_instruct = "Generate an accurate visual description of the image in a single sentence."
        # task_instruct = "Generate a concise and accurate description of the image in one sentence."
        task_instruct = "Describe the main content of the image in one sentence."
        _prompts, _images = [], []
        for img in images:
            _prompts.append(
                [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{task_instruct}",
                        # "images": [example["image"]],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ],
            )
            _images.append([img])

        prepare_inputs = processing_class(
            conversations=_prompts, images=_images, force_batchify=True,
        ).to(device)

        inputs_embeds = cap_mmgpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = cap_mmgpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=1,
            pad_token_id=processing_class.tokenizer.eos_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
        )
        gen_captions = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return gen_captions

    @torch.inference_mode()
    def generate_caption_with_external_model(self, images):
        device = list(self.blip_model.parameters())[0].device

        inputs = self.blip_processor(images=images, return_tensors="pt").to(device)
        out = self.blip_model.generate(**inputs, max_new_tokens=50)
        gen_captions = self.blip_processor.batch_decode(out, skip_special_tokens=True)

        return gen_captions

    @torch.inference_mode()
    def compute_caption_consistency(self, gen_captions, prompts):
        if "bertscore" in self.cap_cs_metrics and "jaccard" in self.cap_cs_metrics:
            if self.using_simcse:
                bert_score_f1 = self.bert_scorer.compute_simcse(gen_captions, prompts)
            else:
                bert_score_f1 = self.bert_scorer.compute_f1(gen_captions, prompts)
            jaccards = [soft_jaccard(pp, gg) for pp, gg in zip(prompts, gen_captions)]
            cap_cs_scores = [(a + b) / 2. for a, b in zip(bert_score_f1, jaccards)]

        elif "jaccard" in self.cap_cs_metrics:
            cap_cs_scores = [soft_jaccard(pp, gg) for pp, gg in zip(prompts, gen_captions)]

        elif "bertscore" in self.cap_cs_metrics:
            if self.using_simcse:
                cap_cs_scores = self.bert_scorer.compute_simcse(gen_captions, prompts)
                cap_cs_scores = [a for a in cap_cs_scores]
            else:
                cap_cs_scores = self.bert_scorer.compute_f1(gen_captions, prompts)
        else:
            raise NotImplementedError("No valid caption consistency computation.")

        return cap_cs_scores

    @torch.inference_mode()
    def compute_image_consistency(self, gen_images, gt_images, device):
        img_cs_scores = []
        for idx, (gen_img, real_img) in enumerate(zip(gen_images, gt_images)):
            if "lpips" in self.img_cs_metrics:
                img_cs_score = 1. - self.lpips_metric(
                    self.transform(gen_img).to(device).unsqueeze(0),
                    self.transform(real_img).to(device).unsqueeze(0)
                )
            else:  # mse
                img_cs_score = 1. - metric_F.image.root_mean_squared_error_using_sliding_window(
                    self.transform(gen_img).to(device).unsqueeze(0),
                    self.transform(real_img).to(device).unsqueeze(0)
                )  # near to 1, the better
            img_cs_scores.append(img_cs_score)

        return img_cs_scores

    def __call__(
            self, completions, prompts, mmgpt=None, processing_class=None, **kwargs
    ):
        device = mmgpt.device

        if self.args.using_external_caption_model:
            gen_captions = self.generate_caption_with_external_model(completions)
        else:
            gen_captions = self.generate_caption_with_policy_mmgpt(
                completions, cap_mmgpt=mmgpt, processing_class=processing_class
            )
        cap_cs_scores = self.compute_caption_consistency(gen_captions, prompts)

        if self.using_img_cs:
            img_cs_scores = self.compute_image_consistency(completions, kwargs['image'], device)

            rewards = [a + b for a, b in zip(cap_cs_scores, img_cs_scores)]
        else:
            rewards = cap_cs_scores

        return rewards

    def load_external_model(self, load_device):
        if self.using_external_caption_model:
            self.blip_processor = BlipProcessor.from_pretrained(f"{self.args.blip_model_ckpt}")
            config = AutoConfig.from_pretrained(f"{self.args.blip_model_ckpt}")
            self.blip_model = BlipForConditionalGeneration(config)
            checkpoint = torch.load(
                f"{self.args.blip_model_ckpt}/pytorch_model.bin", map_location='cpu')
            self.blip_model.load_state_dict(checkpoint, strict=False)
            for param in self.blip_model.parameters():
                param.requires_grad = False

            self.blip_model = self.blip_model.to(load_device)
            self.blip_model = self.blip_model.eval()

        if "bertscore" in self.cap_cs_metrics:
            if self.using_simcse:
                self.bert_scorer = BertSimCSEWrapper(
                    f"{self.args.model_ckpt_dir}/sup-simcse-bert-base-uncased")
                print(f"loaded: sup-simcse-bert-base-uncased")
            else:
                self.bert_scorer = BertScoreWrapper(f"{self.args.model_ckpt_dir}/all-mpnet-base-v2")
                print(f"loaded: all-mpnet-base-v2")

        if self.using_img_cs and "lpips" in self.img_cs_metrics:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg', normalize=True
            ).to(load_device)
            self.lpips_metric = self.lpips_metric.eval()
            print(f"loaded: lpips")

    @property
    def __name__(self):
        return 't2i_CycleConsistency'


class I2TImageCycleConsistencyReward(T2ICycleConsistencyReward):
    def __init__(self, task_args):
        super().__init__(task_args)

    @torch.inference_mode()
    def generate_images_from_captions(self, captions, mmgpt=None, processing_class=None):
        device = mmgpt.device

        prompts = []
        for caption in captions:
            conv = [
                {
                    "role": "<|User|>",
                    "content": caption,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=processing_class.sft_format,
                system_prompt="",
            )
            prompts.append(sft_format + processing_class.image_start_tag)

        prompt_inputs = processing_class.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        _, regenerated_images = mmgpt.t2i_generate_parallel(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            cfg_weight=5,
            parallel_size=1,
            temperature=1,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16,
            pad_id=processing_class.pad_id,
            seed=self.args.seed,
        )
        return regenerated_images

    def _resolve_reference_texts(self, captions=None, detailed_captions=None):
        if captions is None and detailed_captions is None:
            raise ValueError("I2T image cycle reward requires 'caption' or 'detailed_caption'.")

        if captions is None:
            return detailed_captions
        if detailed_captions is None:
            return captions

        refs = []
        for cap, dcap in zip(captions, detailed_captions):
            if cap is not None and str(cap).strip() != "":
                refs.append(cap)
            else:
                refs.append(dcap)
        return refs

    def __call__(
            self, completions, prompts=None, mmgpt=None, processing_class=None, **kwargs
    ):
        device = mmgpt.device

        ref_texts = self._resolve_reference_texts(
            captions=kwargs.get("caption", None),
            detailed_captions=kwargs.get("detailed_caption", None),
        )
        cap_cs_scores = self.compute_caption_consistency(completions, ref_texts)

        if self.using_img_cs:
            regenerated_images = self.generate_images_from_captions(
                completions, mmgpt=mmgpt, processing_class=processing_class
            )
            img_cs_scores = self.compute_image_consistency(
                regenerated_images, kwargs["image"], device
            )
            rewards = [a + b for a, b in zip(cap_cs_scores, img_cs_scores)]
        else:
            rewards = cap_cs_scores

        return rewards

    @property
    def __name__(self):
        return 'i2t_CycleConsistency'


@torch.inference_mode()
def t2i_match_reward(
        completions, prompts=None,
        mmgpt=None, processing_class=None,
        **kwargs
):
    device = mmgpt.device
    _prompts, images = [], []
    for cap, img in zip(prompts, completions):
        _prompts.append(
            [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{cap}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        )
        images.append([img])

    prepare_inputs = processing_class(
        conversations=_prompts, images=images, force_batchify=True,
    ).to(device)

    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask
    )

    all_embeds = outputs.last_hidden_state  # [bs*n, , dim]
    image_embeds = all_embeds[:, 5: 5 + 576]
    text_embeds = all_embeds[:, 586:, :]
    text_masks = prepare_inputs.attention_mask[:, 586:]

    rewards = []
    for image_token, text_token, text_mask in zip(image_embeds, text_embeds, text_masks.bool()):
        rewards.append(
            token_level_max_match_similarity(image_token, text_token[text_mask]).item()
        )
    return rewards

    # mm_sim = metric_F.regression.cosine_similarity(
    #     text_embeds, image_embeds, 'none')  # values in [-1, 1]?

    # # max-min normalize
    # mm_sim = mm_sim.view(-1, num_g)  # [bs, parallel_size]
    # eps = torch.finfo(mm_sim.dtype).eps
    # mm_sim_min = mm_sim.min(dim=1, keepdim=True)[0]
    # mm_sim_max = mm_sim.max(dim=1, keepdim=True)[0]
    # mm_sim = (mm_sim - mm_sim_min) / (mm_sim_max - mm_sim_min + eps)

    # rewards = []
    # for sim in mm_sim:
    #     if sim >= 10.0:
    #         reward = 1.0
    #     elif sim >= 6.0:
    #         reward = 0.5
    #     else:
    #         reward = 0.0
    #     rewards.append(reward)
    # rewards = [sim if sim >= 0 else 0.0 for sim in mm_sim]
    # return rewards


@torch.inference_mode()
def t2i_pixel_mse_reward(
        completions, image,
        processing_class=None,
        **kwargs
):
    gen_images = processing_class.image_processor(completions, return_tensors="pt").pixel_values
    image = processing_class.image_processor(image, return_tensors="pt").pixel_values

    rewards = []
    for gen_img, real_img in zip(gen_images, image):
        reward = 1. - metric_F.image.root_mean_squared_error_using_sliding_window(
            gen_img[None], real_img[None]
        )  # near to 1, the better
        rewards.append(reward)
    return rewards


def classification_accuracy_reward(completions, solution, cap4gen, **kwargs):
    rewards = []
    for idx, (content, sol) in enumerate(zip(completions, solution)):
        content = str(content)
        sol = str(sol)

        reward = 0.0

        # Extract GT answer from solution if it has think/answer tags [always have]
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # word

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)

        if content_match:
            student_answer = content_match.group(1).strip()
            student_answer = student_answer.lower()
            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 1.0

            elif ground_truth in student_answer or student_answer in ground_truth:
                reward = 0.9
            elif student_answer in cap4gen[idx]:
                reward = 0.5

            elif student_answer == "UNK":
                reward = 0.2
        else:
            # remove <think>.*?</think>
            student_answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            student_answer = student_answer.replace(
                '<think>', '').replace('</think>', '').replace(
                '<answer>', '').replace('</answer>', '').replace(
                'Answer:', '').replace('Answer', '').strip()
            student_answer = student_answer.strip().lower()

            # rouge = metric_F.text.rouge.rouge_score(
            #     student_answer, ground_truth, use_stemmer=True, rouge_keys='rougeL'
            # )["recall"]

            if student_answer == ground_truth:
                reward = 0.8
            elif ground_truth in student_answer or student_answer in ground_truth:
                reward = 0.7
            elif student_answer in cap4gen[idx]:
                reward = 0.3

        rewards.append(reward)
    return rewards


@torch.inference_mode()
def t2i_obj_cls_reward(
        completions, prompts=None,
        mmgpt=None, processing_class=None, gen_config=None,
        **kwargs
):
    device = mmgpt.device

    od_prompts, images = [], []
    for que, img in zip(kwargs['cls_problem'], completions):
        od_prompts.append(
            [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{que}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        )
        images.append([img])

    prepare_inputs = processing_class(
        conversations=od_prompts, images=images, force_batchify=True,
    ).to(device)

    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)

    gen_config.num_return_sequences = 1
    outputs = mmgpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        generation_config=gen_config
    )
    answers = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # cls rewards
    rewards = classification_accuracy_reward(
        answers, solution=kwargs['cls_solution'], cap4gen=prompts,
    )
    return rewards


def qa_accuracy_reward(completions, solution, qa_prompts, **kwargs):
    rewards = []
    for idx, (content, sol) in enumerate(zip(completions, solution)):
        content = str(content)
        sol = str(sol)

        reward = 0.0
        # Extract GT answer from solution if it has think/answer tags [always have]
        sol_match = re.search(r"<answer>(.*?)</answer>", sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # A/B/C/D

        # Extract predicted answer from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", content)
        if content_match:
            student_answer = content_match.group(1).strip()
            if student_answer == "UNK":
                reward = 0.2
            # 1) prediction is option letter (A | A.)
            elif len(student_answer) <= 2:
                if safe_string_equal(student_answer, ground_truth):
                    reward = 1.0
            # 2) prediction is <option letter + answer text>, <answer text>, <The answer is xx>
            else:
                if bool(re.search(rf'\b{re.escape(ground_truth)}\b', student_answer)):
                    student_answer = extract_answer_letter_from_response(student_answer)
                    if len(student_answer) <= 2:  # answer letter
                        if safe_string_equal(student_answer, ground_truth):
                            reward = 1.0
                else:  # answer text
                    gt_text = extract_answer_text_from_qa(qa_prompts[idx], ground_truth)
                    if bool(re.search(rf'\b{re.escape(gt_text)}\b', student_answer)):
                        student_answer = extract_answer_text_from_response(student_answer)
                        if safe_string_equal(student_answer, gt_text):
                            reward = 1.0
        else:
            # remove <think>.*?</think>
            student_answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # student_answer = re.sub(r'<think>.*', '', student_answer, flags=re.DOTALL)
            # student_answer = re.sub(r'.*</think>', '', student_answer, flags=re.DOTALL)
            student_answer = student_answer.replace(
                '<think>', '').replace('</think>', '').replace(
                '<answer>', '').replace('</answer>', '').replace(
                'Answer:', '').replace('Answer', '').strip()

            if bool(re.search(rf'\b{re.escape(ground_truth)}\b', student_answer)):
                student_answer = extract_answer_letter_from_response(student_answer)
                if len(student_answer) <= 2:  # answer letter
                    if safe_string_equal(student_answer, ground_truth):
                        reward = 0.8
            else:  # answer text
                gt_text = extract_answer_text_from_qa(qa_prompts[idx], ground_truth)
                if bool(re.search(rf'\b{re.escape(gt_text)}\b', student_answer)):
                    student_answer = extract_answer_text_from_response(student_answer)
                    if safe_string_equal(student_answer, gt_text):
                        reward = 0.8

        rewards.append(reward)
    return rewards


@torch.inference_mode()
def t2i_qa_reward(
        completions, prompts=None,
        mmgpt=None, processing_class=None, gen_config=None,
        **kwargs
):
    device = mmgpt.device

    qa_prompts, images = [], []
    for que, img in zip(kwargs['qa_problem'], completions):
        qa_prompts.append(
            [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{que}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        )
        images.append([img])

    prepare_inputs = processing_class(
        conversations=qa_prompts, images=images, force_batchify=True,
    ).to(device)

    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)

    gen_config.num_return_sequences = 1
    outputs = mmgpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        generation_config=gen_config
    )
    answers = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # qa rewards
    rewards = qa_accuracy_reward(
        answers, solution=kwargs['qa_solution'], qa_prompts=kwargs['qa_problem'],
    )
    return rewards


@torch.inference_mode()
def t2i_clip_reward(
        completions, prompts,
        clip_model=None, clip_tokenizer=None,
        **kwargs
):
    device = completions[0].device

    mean = torch.tensor(OPENAI_DATASET_MEAN, device=device)
    std = torch.tensor(OPENAI_DATASET_STD, device=device)

    image_features = torch.stack(completions, dim=0)
    image_features = image_features / 255.0  # [B, 3, 224, 224]
    image_features = (image_features - mean[None, :, None, None]) / std[None, :, None, None]
    image_features = clip_model.encode_image(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # [B, 1024]

    text_features = clip_tokenizer(prompts).to(device)
    text_features = clip_model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # [B, 1024]

    rewards = []
    for image, text in zip(image_features, text_features):
        similarity = (image @ text.t()).item() * 2.
        rewards.append(similarity)

    return rewards


@torch.inference_mode()
def t2i_cap_consistency_reward(
        completions, prompts,
        caption_model=None, caption_processor=None, bert_scorer=None,
        **kwargs
):
    # completions: PIL.Image
    device = caption_model.device

    # all_images = torch.stack(completions, dim=0)  # float32, 0~255
    inputs = caption_processor(
        images=completions, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs, max_new_tokens=30)
    gen_captions = caption_processor.batch_decode(out, skip_special_tokens=True)

    # rewards = bert_scorer(gen_captions, prompts)['f1'].tolist()

    gts = {}
    res = {}
    for idx, (gen_cap, prompt) in enumerate(zip(gen_captions, prompts)):
        gts[f"{idx}"] = [prompt]
        res[f"{idx}"] = [gen_cap]

    spice_scorer = Spice()
    scores = spice_scorer.compute_score(gts, res)[-1]
    rewards = [a['All']['f'] for a in scores]

    # cider_scorer = Cider()
    # rewards = cider_scorer.compute_score(gts, res)[-1]

    # SPIDEr
    # spide_rscore = 0.5 * cider_score + 0.5 * spice_score

    # meteor_scorer = Meteor()
    # rewards = meteor_scorer.compute_score(gts, res)[-1]

    # evaluator = language_evaluation.CocoEvaluator(
    #     coco_types=["BLEU", "ROUGE_L", "CIDEr", "SPICE"]
    # )
    # results = evaluator.run_evaluation(gen_captions, prompts)

    return rewards


# from .r_od import (
#     accuracy_reward_iou,
#     accuracy_reward_confidence,
# )
@torch.inference_mode()
def t2i_obj_det_reward(
        completions,
        mmgpt=None, processing_class=None, gen_config=None,
        **kwargs
):
    device = mmgpt.device

    od_prompts, images = [], []
    for que, img in zip(kwargs['od_problem'], completions):
        od_prompts.append(
            [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{que}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        )
        images.append([img])

    prepare_inputs = processing_class(
        conversations=od_prompts, images=images, force_batchify=True,
    ).to(device)

    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)

    gen_config.num_return_sequences = 1
    outputs = mmgpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        generation_config=gen_config
    )
    answers = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # od rewards
    rewards = []
    return rewards
