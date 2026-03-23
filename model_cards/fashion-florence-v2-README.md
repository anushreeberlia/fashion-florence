# Fashion Florence V2: 8-Field Fashion Tagging with Florence-2

A LoRA adapter for Florence-2 that analyzes clothing images and returns structured fashion tags as JSON with 8 fields — extending [Fashion Florence V1](https://huggingface.co/anushreeberlia/fashion-florence) from 4 to 8 attributes.

## Model Summary

| | |
|---|---|
| **Base model** | [anushreeberlia/fashion-florence](https://huggingface.co/anushreeberlia/fashion-florence) (Florence-2-large, 0.77B) |
| **Method** | LoRA adapter (r=16, alpha=32), 62MB |
| **Task** | Clothing image → structured JSON fashion tags (8 fields) |
| **Training data** | 17,640 images from [Marqo/iMaterialist](https://huggingface.co/datasets/Marqo/iMaterialist), labeled by GPT-4o-mini |
| **License** | MIT |

## Output Schema

| Field | Type | Model | Example |
|---|---|---|---|
| category | string | predicted | `"dress"` |
| primary_color | string | predicted | `"pink"` |
| secondary_colors | list | predicted | `["black"]` |
| material | string | predicted | `"lace"` |
| fit | string | predicted | `"bodycon"` |
| style_tags | list | predicted | `["sexy", "glamorous", "trendy"]` |
| occasion_tags | list | post-processed | `["party", "night-out"]` |
| season_tags | list | post-processed | `["spring", "summer", "fall"]` |

The model reliably predicts the first 6 fields directly from the image. `occasion_tags` and `season_tags` are derived from the predicted fields using rule-based post-processing (see [inference.py](https://github.com/anushreeberlia/fashion-florence/blob/main/training/inference.py)).

## How to Use

### With post-processing (recommended)

```python
import torch, json
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "anushreeberlia/fashion-florence"
ADAPTER = "anushreeberlia/fashion-florence-v2"

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER).eval().to("cuda")

image = Image.open("clothing.jpg").convert("RGB")
prompt = "Analyze this clothing item image and return structured fashion tags as JSON."
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, num_beams=1, do_sample=False)
raw = processor.tokenizer.decode(out[0], skip_special_tokens=True)

# Post-process: parse JSON + infer occasion/season from predicted fields
from training.inference import predict_with_postprocess
tags = predict_with_postprocess(raw)
print(json.dumps(tags, indent=2))
```

### Output example

```json
{
  "category": "dress",
  "primary_color": "pink",
  "secondary_colors": [],
  "material": "lace",
  "fit": "bodycon",
  "style_tags": ["sexy", "glamorous", "trendy"],
  "occasion_tags": ["everyday", "going-out", "night-out", "party"],
  "season_tags": ["spring", "summer", "fall"]
}
```

## V1 vs V2

| | V1 | V2 |
|---|---|---|
| Output fields | 4 | 8 (6 predicted + 2 inferred) |
| Training labels | Rule-mapped from iMaterialist metadata | GPT-4o-mini vision labels |
| New fields | — | secondary_colors, fit, occasion_tags, season_tags |
| Model size | 0.77B (merged) | 0.77B base + 62MB adapter |
| Category accuracy | 89.5% | ~90% |

## Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Effective batch size | 16 (2 × 8 gradient accumulation) |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |
| Precision | fp16 |
| Max target length | 512 tokens |
| Training examples | 17,640 |
| Validation examples | 980 |
| Final eval loss | 0.088 |
| Training time | ~3 hours on A100 40GB |

## Training Data

Labels were generated using GPT-4o-mini as a teacher model (model distillation). Each image from the iMaterialist dataset was processed through GPT-4o-mini vision to produce structured 8-field JSON annotations. Images are from a non-overlapping slice of the dataset (offset 50,000+) to avoid overlap with V1 training data.

## Limitations

- `occasion_tags` and `season_tags` are rule-inferred, not predicted by the model
- Performance on shoes and accessories is lower due to limited training examples in those categories
- The model works best on single-garment product images against clean backgrounds

## Citation

```bibtex
@article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```
