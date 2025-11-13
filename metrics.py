import os
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from tqdm import tqdm
import yaml

def calculate_dino_style_similarity(
    prompt_path,
    style_dir,
    method_dirs,
    save_yaml_path,
    model_path="dino-vitb8"
):
    device = "cuda"
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTModel.from_pretrained(model_path).to(device)

    with open(prompt_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    num_prompts = len(prompts)

    style_images = sorted([f for f in os.listdir(style_dir) if f.endswith(".jpg") and f != "0000.jpg"])

    if os.path.exists(save_yaml_path):
        with open(save_yaml_path, 'r') as f:
            all_results = yaml.safe_load(f) or {}
    else:
        all_results = {}

    for method_dir in method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\n[INFO] Processing method: {method_name}")

        total_score = 0.0
        total_image_count = 0

        for style_img in tqdm(style_images, desc=f"{method_name} - Style Images"):
            style_path = os.path.join(style_dir, style_img)
            image = Image.open(style_path).convert("RGB")
            style_feat = model(**processor(images=image, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]

            for i, prompt in enumerate(prompts):
                result_path = os.path.join(method_dir, f"prompt-{i+1}", style_img)
                if not os.path.exists(result_path):
                    continue

                image_gen = Image.open(result_path).convert("RGB")
                gen_feat = model(**processor(images=image_gen, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]

                sim = F.cosine_similarity(style_feat, gen_feat, dim=-1).item()
                total_score += sim
                total_image_count += 1

        avg_score = total_score / total_image_count if total_image_count > 0 else 0.0
        print(f"[RESULT] {method_name} | DINO Style Score: {avg_score:.4f}")
        all_results[method_name] = round(avg_score, 4)

    os.makedirs(os.path.dirname(save_yaml_path), exist_ok=True)
    with open(save_yaml_path, "w") as f:
        yaml.dump(all_results, f)
    print(f"[INFO] All results saved to: {save_yaml_path}")

def calculate_clip_style_scores(
    prompt_path,
    style_dir,
    method_dirs,
    save_yaml_path,
    model_path="clip-vit-large-patch14"
):
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPImageProcessor.from_pretrained(model_path)

    with open(prompt_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    num_prompts = len(prompts)

    style_images = sorted([f for f in os.listdir(style_dir) if f.endswith('.jpg') and f != "0000.jpg"])

    if os.path.exists(save_yaml_path):
        with open(save_yaml_path, 'r') as f:
            all_results = yaml.safe_load(f) or {}
    else:
        all_results = {}

    for method_dir in method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\n[INFO] Processing method: {method_name}")

        total_score = 0.0
        total_count = 0

        for style_img in tqdm(style_images, desc=f"{method_name} - Style Loop"):
            style_path = os.path.join(style_dir, style_img)
            style_feat = model.get_image_features(
                processor(images=Image.open(style_path).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
            )

            for i in range(num_prompts):
                result_img_path = os.path.join(method_dir, f"prompt-{i+1}", style_img)
                if not os.path.exists(result_img_path):
                    print("[Warning] wrong file wrong file!!!")
                    continue

                result_feat = model.get_image_features(
                    processor(images=Image.open(result_img_path).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
                )

                sim = F.cosine_similarity(style_feat, result_feat).item()
                total_score += sim
                total_count += 1

        avg_score = total_score / total_count if total_count > 0 else 0.0
        print(f"[RESULT] {method_name} | CLIP Style Score: {avg_score:.4f}")
        all_results[method_name] = round(avg_score, 4)

    os.makedirs(os.path.dirname(save_yaml_path), exist_ok=True)
    with open(save_yaml_path, "w") as f:
        yaml.dump(all_results, f)
    print(f"[INFO] All results saved to: {save_yaml_path}")


def calculate_clip_text_similarity(
    prompt_path,
    method_dirs,
    save_yaml_path,
    model_path="clip-vit-large-patch14/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    if os.path.exists(save_yaml_path):
        with open(save_yaml_path, 'r') as f:
            all_results = yaml.safe_load(f) or {}
    else:
        all_results = {}

    for method_dir in method_dirs:
        method_name = os.path.basename(method_dir)
        print(f"\n[INFO] Processing method: {method_name}")
        total_score = 0.0
        total_count = 0

        for i, prompt in enumerate(tqdm(prompts, desc="Prompt folders")):
            prompt_folder = os.path.join(method_dir, f"prompt-{i+1}")
            if not os.path.isdir(prompt_folder):
                print(f"[WARN] Missing: {prompt_folder}")
                continue

            for image_name in sorted(os.listdir(prompt_folder)):
                if not image_name.endswith(".jpg"):
                    continue
                image_path = os.path.join(prompt_folder, image_name)
                image = Image.open(image_path).convert("RGB")

                image_inputs = processor(images=image, return_tensors="pt").to(device)
                text_inputs = processor(text=prompt, return_tensors="pt").to(device)

                image_feat = model.get_image_features(**image_inputs)
                text_feat = model.get_text_features(**text_inputs)
                sim = F.cosine_similarity(image_feat, text_feat).item()

                total_score += sim
                total_count += 1

        avg_score = total_score / total_count if total_count > 0 else 0.0
        print(f"[RESULT] {method_name} | Avg CLIP Text Similarity: {avg_score:.4f}")
        all_results[method_name] = round(avg_score, 4)

    os.makedirs(os.path.dirname(save_yaml_path), exist_ok=True)
    with open(save_yaml_path, "w") as f:
        yaml.dump(all_results, f)
    print(f"[INFO] Saved to {save_yaml_path}")

import matplotlib.pyplot as plt
def plot_metric_comparison(clip_ta_yaml, clip_ss_yaml, dino_ss_yaml, output_path='metric_comparison.png'):
    with open(clip_ta_yaml, 'r') as f:
        clip_ta = yaml.safe_load(f)
    with open(clip_ss_yaml, 'r') as f:
        clip_ss = yaml.safe_load(f)
    with open(dino_ss_yaml, 'r') as f:
        dino_ss = yaml.safe_load(f)

    keys = sorted(clip_ta.keys())
    x_labels = [chr(97 + i) for i in range(len(keys))]  # a, b, c, ...

    clip_ta_values = [clip_ta[k] for k in keys]
    clip_ss_values = [clip_ss[k] for k in keys]
    dino_ss_values = [dino_ss[k] for k in keys]

    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, clip_ta_values, marker='o', label='CLIP Text Alignment')
    plt.plot(x_labels, clip_ss_values, marker='s', label='CLIP Style Similarity')
    plt.plot(x_labels, dino_ss_values, marker='^', label='DINO Style Similarity')

    # Add numeric labels above each point
    for x, y in zip(x_labels, clip_ta_values):
        plt.text(x, y + 0.002, f"{y:.4f}", ha='center', va='bottom', fontsize=8)
    for x, y in zip(x_labels, clip_ss_values):
        plt.text(x, y + 0.002, f"{y:.4f}", ha='center', va='bottom', fontsize=8)
    for x, y in zip(x_labels, dino_ss_values):
        plt.text(x, y + 0.002, f"{y:.4f}", ha='center', va='bottom', fontsize=8)

    plt.xlabel('Methods')
    plt.ylabel('Score')
    plt.title('Metric Comparison Across Methods')
    plt.legend()
    plt.grid(True)

    plt.xticks(x_labels)
    plt.subplots_adjust(bottom=0.3)
    for i, key in enumerate(keys):
        plt.text(i, plt.ylim()[0] - 0.02 * (plt.ylim()[1] - plt.ylim()[0]),
                 key, rotation=45, ha='center', va='top', fontsize=8)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" Metric comparison plot saved to {output_path}")

def main(prompt_path, style_dir, method_dirs,
         clip_ta_yaml, clip_ss_yaml, dino_ss_yaml,
         metric_visual_path,):
    
    calculate_clip_text_similarity(
        prompt_path=prompt_path,
        method_dirs=method_dirs,
        save_yaml_path=clip_ta_yaml,
    )
    
    calculate_clip_style_scores(
        prompt_path=prompt_path,
        style_dir=style_dir,
        method_dirs=method_dirs,
        save_yaml_path=clip_ss_yaml,
    )
    
    calculate_dino_style_similarity(
        prompt_path=prompt_path,
        style_dir=style_dir,
        method_dirs=method_dirs,
        save_yaml_path=dino_ss_yaml,
    )
    
    # TODO:
    # plot_metric_comparison(clip_ta_yaml, clip_ss_yaml, dino_ss_yaml, metric_visual_path)

if __name__ == "__main__":
    prompt_path = "prompts.txt"
    style_dir = "style_images"
    clip_ta_yaml = "clip_text_scores.yaml"
    clip_ss_yaml = "clip_style_scores.yaml"
    dino_ss_yaml = "dino_style_scores.yaml"
    method_dirs = [ "gen_dir"],
    metric_visual_path = "metric_comparison.png"
    main(prompt_path, style_dir, method_dirs,
         clip_ta_yaml=clip_ta_yaml,
         clip_ss_yaml=clip_ss_yaml,
         dino_ss_yaml=dino_ss_yaml,
         metric_visual_path=metric_visual_path,)
