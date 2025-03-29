#import torch
#from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

# `single_file_url` could also be https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors
# single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
# transformer = LTXVideoTransformer3DModel.from_single_file(
#   single_file_url, torch_dtype=torch.bfloat16
# )
# vae = AutoencoderKLLTXVideo.from_single_file(single_file_url, torch_dtype=torch.bfloat16)
# pipe = LTXImageToVideoPipeline.from_pretrained(
#   "Lightricks/LTX-Video", transformer=transformer, vae=vae, torch_dtype=torch.bfloat16#
# )

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Lightricks/LTX-Video", filename="ltx-video-2b-v0.9.safetensors", local_dir="/models")

