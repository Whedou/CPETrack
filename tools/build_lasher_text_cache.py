import argparse
import os
import sys

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.utils.text_semantic import clean_sequence_name


def read_sequence_names(seq_list_paths, dataset_roots=None):
    names = []

    for seq_list_path in seq_list_paths:
        if not seq_list_path:
            continue
        with open(seq_list_path, "r") as f:
            names.extend(line.strip() for line in f if line.strip())

    for dataset_root in dataset_roots or []:
        if not dataset_root:
            continue
        names.extend(
            name for name in sorted(os.listdir(dataset_root))
            if os.path.isdir(os.path.join(dataset_root, name))
        )

    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return unique_names


def load_open_clip(model_name, pretrained, device):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    def encode(texts):
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens)
        return text_features

    return encode


def load_clip(model_name, device):
    import clip

    model, _ = clip.load(model_name, device=device)
    model.eval()

    def encode(texts):
        tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(tokens)
        return text_features

    return encode


def build_features(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seq_names = read_sequence_names(args.seq_list, args.dataset_root)
    if not seq_names:
        raise RuntimeError("No sequence names found. Check --seq-list or --dataset-root.")

    provider_errors = []
    encode_text = None

    if args.provider in ("auto", "open_clip"):
        try:
            encode_text = load_open_clip(args.open_clip_model, args.open_clip_pretrained, device)
            print("Using open_clip {} ({})".format(args.open_clip_model, args.open_clip_pretrained))
        except Exception as e:
            provider_errors.append("open_clip: {}".format(e))
            if args.provider == "open_clip":
                raise

    if encode_text is None and args.provider in ("auto", "clip"):
        try:
            encode_text = load_clip(args.clip_model, device)
            print("Using clip {}".format(args.clip_model))
        except Exception as e:
            provider_errors.append("clip: {}".format(e))
            if args.provider == "clip":
                raise

    if encode_text is None:
        raise RuntimeError(
            "Could not load a CLIP text encoder.\n"
            + "\n".join(provider_errors)
            + "\nInstall open_clip_torch or clip in the training environment, "
              "or make sure the model weights are already cached locally."
        )

    features = {}
    prompts = []
    prompt_to_names = []
    for seq_name in seq_names:
        clean_name = clean_sequence_name(seq_name)
        text = clean_name if clean_name else seq_name
        prompt = args.prompt_template.format(text)
        prompts.append(prompt)
        prompt_to_names.append((seq_name, clean_name))

    for start in range(0, len(prompts), args.batch_size):
        end = start + args.batch_size
        batch_prompts = prompts[start:end]
        batch_names = prompt_to_names[start:end]
        with torch.no_grad():
            batch_features = encode_text(batch_prompts).float()
            if args.normalize:
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            batch_features = batch_features.cpu()

        for feature, (seq_name, clean_name) in zip(batch_features, batch_names):
            features[seq_name.lower()] = feature
            if clean_name:
                features[clean_name] = feature

        print("Encoded {}/{} sequence names".format(min(end, len(prompts)), len(prompts)))

    feature_dim = int(next(iter(features.values())).numel())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "feature_dim": feature_dim,
            "prompt_template": args.prompt_template,
            "features": features,
        },
        args.output,
    )
    print("Saved {} text features to {}".format(len(features), args.output))
    print("feature_dim:", feature_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="Build LasHeR sequence-name text feature cache.")
    parser.add_argument(
        "--seq-list",
        nargs="*",
        default=[
            "/data/pudata/LasHeR/TrainingSet/train/trainingsetList.txt",
            "/data/pudata/LasHeR/TestingSet/testingset/testingsetList.txt",
        ],
        help="Text files containing sequence names.",
    )
    parser.add_argument(
        "--dataset-root",
        nargs="*",
        default=[],
        help="Optional dataset roots whose immediate subdirectories are sequence names.",
    )
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "cache", "lasher_text_features.pt"))
    parser.add_argument("--provider", choices=["auto", "open_clip", "clip"], default="open_clip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prompt-template", default="the target object is {}")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--open-clip-model", default="ViT-B-32")
    parser.add_argument("--open-clip-pretrained", default="openai")
    parser.add_argument("--clip-model", default="ViT-B/32")
    return parser.parse_args()


if __name__ == "__main__":
    build_features(parse_args())
