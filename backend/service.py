import torch
import datetime
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.models import AutoencoderKL


# モデル設定
# https://huggingface.co/stablediffusionapi/anything-v5
# 自由にDiffusionモデルを変更してください
# ただしライセンスには注意してください
model_id = "stablediffusionapi/anything-v5"


# VAEの設定
# https://huggingface.co/xyn-ai/anything-v4.0/tree/main
# anything-v4.0.vae.ptをLFSボタンでダウンロードしてください。
# VAEは変換が必要なのでhttps://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.pyのコードを使って変換してください。
# python ./convert_vae_pt_to_diffusers.py --vae_pt_path ./vae/anything-v4.0.vae.pt --dump_path ./vae/anythingv4_vaeを実行してください。
vae_id = "./vae/anythingv4_vae"

# GPUチェック
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# モデルの読み込み
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# VAEの読み込み
# pipe.vae = AutoencoderKL.from_pretrained(vae_id)


# スケジューラーの設定
# ノイズスケジューラはいくつかあるので試してみてください
# https://huggingface.co/docs/diffusers/api/schedulers/overview
# pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# デバイスの設定
pipe = pipe.to(device)


# 画像生成
async def generate_image(payload):
    generator_list = []

    # シードを設定
    for i in range(payload.count):
        generator_list.append(torch.Generator(device).manual_seed(payload.seedList[i]))

    # 画像生成
    images_list = pipe(
        [payload.prompt] * payload.count,
        width=payload.width,
        height=payload.height,
        negative_prompt=[payload.negative] * payload.count,
        guidance_scale=payload.scale,
        num_inference_steps=payload.steps,
        generator=generator_list,
    )

    images = []
    # 画像を保存
    for i, image in enumerate(images_list["images"]):
        file_name = (
            "./outputs/image_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            + ".png"
        )
        image.save(file_name)
        images.append(image)

    return images