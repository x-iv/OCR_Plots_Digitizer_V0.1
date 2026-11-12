# OCR_Plots_Digitizer_V0.1
A novel Vision-Language Model pipeline for digitizing complex scientific plots into structured JSON. V0.1 features a 2B parameter model trained with Reinforcement Learning to internally apply visual processing techniques (MoVE) and extract precise data without external OpenCV pipelines.

## 📖 Overview
Scientific charts are the primary medium for data communication, but the underlying raw data is rarely published. Current OCR models struggle with complex scientific plots (multi-axis, error bars, overlapping series). 

**OCR Plots Digitizer** is a specialized Vision-Language Model (VLM) built on `Qwen2-VL-2B-Instruct`. It uses a novel **Mixture of Visual Experts (MoVE)** architecture. Instead of relying on external Python scripts to clean images, the model is trained to internally apply visual reasoning tags (`<expert>CLAHE</expert>`, `<expert>Denoise</expert>`) to adjust its own attention weights, digitizing messy charts directly into structured JSON.

## 🧠 Training Methodology (V0.1)

The V0.1 model was trained using a two-stage pipeline on a dataset of synthetically generated arXiv-style plots:

1. **Synthetic Data Engine:** Used DeepSeek LLM to generate complex JSON schemas and Matplotlib rendering code, paired with random visual degradations (JPEG compression, noise).
2. **SFT Warmup:** Supervised Fine-Tuning to teach the model the strict `<thought>`, `<expert>`, and `<json>` syntax.
3. **GRPO (Reinforcement Learning):** Group Relative Policy Optimization to optimize three specific reward functions:
   * **MoVE Routing Reward:** Rewards the model for correctly identifying the visual degradation and selecting the right internal expert (CLAHE, Denoise, Edge, Otsu, Morphology).
   * **Continuous MAE Reward:** Uses an exponential reward based on Mean Absolute Error to break the model's habit of quantizing floats (outputting `0.78` instead of `0.8`).
   * **Variance Reward:** Forces the model to flatten its probability distribution, natively embedding Gaussian sensor noise into the output.

## 📊 V0.1 Results & Evaluation

The model was evaluated on a completely unseen **10% holdout test set** (98 images). 

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **JSON Validity Rate** | **89.8%** | Successfully generated parsable JSON. |
| **MoVE Routing Accuracy**| **100%** | Perfectly matched the internal expert to the visual degradation. |
| **Data Extraction Rate** | **89.8%** | Successfully mapped pixel coordinates to `x` and `y` arrays. |

### 🔬 Visual Verification
The script generates overlays to compare the Ground Truth (Blue Circles) against the Digitized Output (Red Crosses). 
*(Insert a screenshot of one of your best verification plots here)*

### ⚠️ Known Limitations (V0.1)
* **Dual Y-Axes Confusion:** The model occasionally fails on plots with dual Y-axes, sometimes flatlining secondary lines to `0`.
* **Continuous Line Regression:** While excellent at discrete scatter points with error bars, the model struggles slightly with continuous theoretical curves without markers.
* **Spatial Quantization:** Being a 2B parameter model, it occasionally outputs `0.8` instead of `0.783` for extremely dense heatmaps.

## 🚀 How to Use

You can load the model directly from Hugging Face:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_id = "xivfrank/OCR_Plots_Digitizer_V0.1"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("your_chart.png").convert("RGB")
prompt = "<image>\nExtract data. Select best expert from ['CLAHE', 'Bilateral_NoiseReduction', 'Canny_EdgeDetection', 'Otsu_Thresholding', 'Morphology_Clean', 'None']. Act as a physical sensor with Gaussian measurement error."

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
inputs = processor(image, processor.apply_chat_template(messages, add_generation_prompt=True), return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
print(processor.decode(outputs[0], skip_special_tokens=True))
