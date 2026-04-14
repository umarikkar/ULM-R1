"""
Sanity-check Janus-Pro-1B's out-of-the-box image-to-text behavior on a few
query images from the VL-Health Instruct Fine-Tuning / Comprehension data.

We don't ask for detailed captions — just a short "what does this image
represent?" so we can eyeball whether the base model recognizes the modality
/ subject matter at all.

Usage (inside the corl env):
    python test_janus_medical_i2t.py
    python test_janus_medical_i2t.py --num-images 8 --max-new-tokens 128
"""

import argparse
import io
import os
import time

import pyarrow.parquet as pq
import torch

# cuDNN in this conda env fails to initialize on conv2d
# (CUDNN_STATUS_NOT_INITIALIZED). Disable cuDNN so convs fall back to
# native CUDA kernels. Must be set before any conv runs.
torch.backends.cudnn.enabled = False
from PIL import Image
from transformers import AutoModelForCausalLM

from janus.models import VLChatProcessor


QUESTION = "<image_placeholder>\nWhat does this image represent? Answer in one short sentence."


@torch.inference_mode()
def answer_one(vl_gpt, vl_chat_processor, pil_image: Image.Image,
               max_new_tokens: int = 96) -> str:
    conversation = [
        {
            "role": "<|User|>",
            "content": QUESTION,
            "images": [pil_image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
    ).to(vl_gpt.device)

    # cast pixel values to the model's dtype (bf16)
    prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(
        next(vl_gpt.parameters()).dtype
    )

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    answer = vl_chat_processor.tokenizer.decode(
        outputs[0].cpu().tolist(), skip_special_tokens=True
    )
    # This tokenizer leaves GPT-2-style BPE "Ġ" space markers in the output;
    # convert them back to real spaces.
    answer = answer.replace("\u0120", " ").replace("Ġ", " ")
    return answer.strip()


def load_query_images(parquet_path: str, n: int):
    """Return list of (image_id, PIL.Image, gt_first_turn_text)."""
    t = pq.read_table(parquet_path)
    # Stride across the shard so we don't get N near-duplicates from the top.
    # pyarrow's .take() concatenates chunks and overflows on the binary
    # `images` column (>2GB), so slice 1-row windows instead.
    stride = max(1, t.num_rows // max(n, 1))

    out = []
    for i in range(n):
        row = t.slice(i * stride, 1).to_pydict()
        img = Image.open(io.BytesIO(row["images"][0])).convert("RGB")
        convs = row["conversations"][0]
        gt_user = convs[0]["value"] if convs else ""
        out.append((row["image_ids"][0], img, gt_user))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument(
        "--parquet",
        type=str,
        default="/work/um00109/MLLM/datasets/VL-Health/Instruct_Fine_Tuning/"
                "Comprehension/train-00000-of-00009.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/work/um00109/MLLM/datasets/VL-Health/janus_medical_i2t_samples",
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "queries")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Loading VLChatProcessor from {args.model_path}")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    print(f"Loading MultiModalityCausalLM from {args.model_path}")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print(f"Model ready on {next(vl_gpt.parameters()).device}")

    print(f"Loading {args.num_images} query images from {args.parquet}")
    samples = load_query_images(args.parquet, args.num_images)

    log_path = os.path.join(args.out_dir, "answers.txt")
    total_t0 = time.time()

    with open(log_path, "w") as f:
        for idx, (img_id, img, gt_user) in enumerate(samples):
            img.save(os.path.join(img_dir, f"{idx:02d}_{img_id}"))

            t0 = time.time()
            answer = answer_one(
                vl_gpt, vl_chat_processor, img,
                max_new_tokens=args.max_new_tokens,
            )
            dt = time.time() - t0

            header = f"[{idx:02d}] {img_id}  ({img.size[0]}x{img.size[1]})  {dt:.1f}s"
            print("\n" + header)
            print(f"  Q: {QUESTION.replace(chr(10), ' ')}")
            print(f"  A: {answer}")
            # Show the dataset's own first-turn user question for context —
            # it usually hints at modality / anatomy.
            gt_preview = gt_user.replace("\n", " ")[:140]
            print(f"  (dataset asked: {gt_preview})")

            f.write(header + "\n")
            f.write(f"  Q: {QUESTION}\n")
            f.write(f"  A: {answer}\n")
            f.write(f"  (dataset asked: {gt_preview})\n\n")

    print(f"\nAll done in {time.time() - total_t0:.0f}s")
    print(f"Images:  {img_dir}")
    print(f"Answers: {log_path}")


if __name__ == "__main__":
    main()
