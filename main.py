import os
import urllib.request
import cv2
import numpy as np
import torch
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
SEGMENTER_PATH = "hair_segmenter.tflite"
SD_MODEL_ID = "Uminosachi/realisticVisionV51_v51VAE-inpainting"

HAIRSTYLES_DB = {
    "Мужская": {
        "Короткие": {
            "Текстурированный Кроп": "A detailed portrait of a man, short textured crop haircut, sharp hair contours, highly detailed hair strands, depth of field, photorealistic, 8k",
            "Классический Андеркат": "A portrait of a man with a sharp undercut hairstyle, combed back hair, visible hair texture, clean sides, realistic lighting",
            "Спортивный Бокс": "A man with a very short classic buzzcut haircut, realistic detailed hair stubble, clean masculine look, sharp focus"
        },
        "Длинные": {
            "Мужские кудри средней длины": "A man with medium-length curly hair, natural dark ringlets, perfect curl definition, highly detailed strands, realistic hair volume",
            "Длинные мужественные волны": "A man with shoulder-length flowing wavy hair, textured masculine waves, highly detailed hair texture, natural daylight",
            "Мужской пучок (Man Bun)": "A portrait of a man with long hair tied into a neat top knot, man bun, shaved sides, highly detailed hair texture"
        }
    },
    "Женская": {
        "Короткие": {
            "Стильное Каре (Bob cut)": "A gorgeous woman, elegant straight bob haircut, dark brunette hair, silky smooth texture, reflective shine, sharp symmetrical edges, 8k",
            "Дерзкая Пикси (Pixie)": "A woman with a modern short pixie haircut, textured layers, highly detailed individual hair strands, professional studio lighting",
            "Короткие кудри с объемом": "A woman with short voluminous curly hair, tight natural ringlets, high quality detailed texture, realistic shape"
        },
        "Длинные": {
            "Голливудские светлые локоны": "A beautiful woman, long flowing wavy blonde hair, luxury Hollywood volume, highly detailed hair texture, realistic strands",
            "Длинные прямые темные волосы": "A woman with very long straight silky brunette hair, sleek look, reflective hair texture, flawless realistic strands",
            "Пышные рыжие кудри": "A woman with long voluminous ginger curly hair, vibrant natural red color, highly detailed ringlets, realistic volume"
        }
    }
}

print("Запуск ИИ-модулей...")
if not os.path.exists(SEGMENTER_PATH):
    urllib.request.urlretrieve(MODEL_URL, SEGMENTER_PATH)

FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
default_steps = 50 if device == "cuda" else 20

pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch_dtype).to(device)
pipe.safety_checker = None
pipe.requires_safety_checker = False

base_options = python.BaseOptions(model_asset_path=SEGMENTER_PATH)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options)

print(f"Успешно! Активный девайс: {device.upper()}")



def create_dual_subtracted_mask(raw_hair_mask, orig_bgr, length):
    h, w = raw_hair_mask.shape[:2]

    # Маска 1: Волосы
    hair_layer = raw_hair_mask.copy()
    if length == "Длинные":
        kernel_vol = np.ones((25, 25), np.uint8)
        hair_layer = cv2.dilate(hair_layer, kernel_vol, iterations=1)

        gray_temp = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        faces_temp = face_detector.detectMultiScale(gray_temp, 1.1, 5, minSize=(100, 100))
        if len(faces_temp) > 0:
            fx, fy, fw, fh = max(faces_temp, key=lambda b: b[2] * b[3])

            cv2.rectangle(hair_layer, (max(0, fx - int(fw * 0.5)), fy + int(fh * 0.4)), (fx + int(fw * 0.2), h), 255,
                          -1)
            cv2.rectangle(hair_layer, (fx + fw - int(fw * 0.2), fy + int(fh * 0.4)),
                          (min(w, fx + fw + int(fw * 0.5)), h), 255, -1)

    # Маска 2: Вычитание
    face_layer = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        center_x = fx + fw // 2
        # Строим плотный защитный овал лица (захватывает лоб, глаза, щеки, подбородок)
        axes = (int(fw * 0.42), int(fh * 0.62))
        cv2.ellipse(face_layer, (center_x, fy + int(fh * 0.52)), axes, 0, 0, 360, 255, -1)

    final_subtracted_mask = cv2.subtract(hair_layer, face_layer)
    return final_subtracted_mask


def get_crop_coordinates(binary_mask, padding=140):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    h_img, w_img = binary_mask.shape[:2]
    return max(0, x - padding), max(0, y - padding), min(w_img, x + w + padding), min(h_img, y + h + padding)


def process_mask_for_pass(mask_bgr, pass_idx):
    mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    if pass_idx == 1:
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(thresh, kernel, iterations=1)
        return cv2.GaussianBlur(mask_dilated, (21, 21), 0)
    else:
        kernel_outer = np.ones((13, 13), np.uint8)
        kernel_inner = np.ones((7, 7), np.uint8)
        dilation = cv2.dilate(thresh, kernel_outer, iterations=1)
        erosion = cv2.erode(thresh, kernel_inner, iterations=1)
        edge_mask = cv2.subtract(dilation, erosion)
        blur_size = (11, 11) if pass_idx == 3 else (15, 15)
        return cv2.GaussianBlur(edge_mask, blur_size, 0)


def merge_step(crop_512, base_high_res_rgb, coords, pass_mask_gray):
    x1, y1, x2, y2 = coords
    crop_w, crop_h = (x2 - x1), (y2 - y1)
    crop_native = crop_512.resize((crop_w, crop_h), resample=Image.Resampling.LANCZOS)
    crop_np = np.array(crop_native)

    final_crop_mask_gray = cv2.resize(pass_mask_gray, (crop_w, crop_h)) / 255.0
    final_crop_mask = np.expand_dims(final_crop_mask_gray, axis=2)

    bg_part = base_high_res_rgb[y1:y2, x1:x2] * (1.0 - final_crop_mask)
    fg_part = crop_np * final_crop_mask
    combined_crop = (bg_part + fg_part).astype(np.uint8)

    updated_high_res = base_high_res_rgb.copy()
    updated_high_res[y1:y2, x1:x2] = combined_crop
    return updated_high_res


def generate_web_interface(input_image_path, gender, length, style_name, custom_user_prompt):
    if input_image_path is None or not style_name:
        return None, None, None

    orig_bgr = cv2.imread(input_image_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)


    mp_image = mp.Image.create_from_file(input_image_path)
    segmentation_result = segmenter.segment(mp_image)
    raw_binary_mask = np.where(segmentation_result.category_mask.numpy_view() == 1, 255, 0).astype(np.uint8)


    working_binary_mask = create_dual_subtracted_mask(raw_binary_mask, orig_bgr, length)

    coords = get_crop_coordinates(working_binary_mask)
    if coords is None: return None, None, None
    x1, y1, x2, y2 = coords

    crop_img_rgb = orig_rgb[y1:y2, x1:x2]
    crop_mask_bgr = cv2.cvtColor(working_binary_mask[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
    init_image_512 = Image.fromarray(crop_img_rgb).resize((512, 512), resample=Image.Resampling.LANCZOS)

    # Сборка промпта
    if style_name == "Свой вариант":
        print(f"Сборка кастомного промпта пользователя...")
        gender_prefix = "A professional detailed portrait of a man" if gender == "Мужская" else "A professional detailed portrait of a woman"
        length_desc = "short modern hair" if length == "Короткие" else "long flowing beautiful hair"

        selected_prompt = f"{gender_prefix}, {length_desc}, {custom_user_prompt}, clean hair texture, highly detailed hair strands, no artifacts, beautiful look, realistic lighting, photorealistic, 8k, highly focused"
    else:
        selected_prompt = HAIRSTYLES_DB[gender][length][style_name]

    base_negative = "bad anatomy, deformed, ugly, blurry, low quality, noise, artifacts, cartoon, 3d render, text, watermark, hairpin, hair clip, hair accessory, bobby pin, jewelry, headband, tiara, hair decoration"
    current_canvas = orig_rgb.copy()


    print(f"[ПРОХОД 1/3] Перестройка структуры волос...")
    p1_mask_gray = process_mask_for_pass(crop_mask_bgr, 1)
    p1_mask_pil = Image.fromarray(p1_mask_gray).resize((512, 512), resample=Image.Resampling.NEAREST)
    output_pass1 = \
    pipe(prompt=selected_prompt, negative_prompt=base_negative, image=init_image_512, mask_image=p1_mask_pil,
         num_inference_steps=default_steps, guidance_scale=9.5, strength=0.95).images[0]
    res1_np = merge_step(output_pass1, current_canvas, coords, p1_mask_gray)

    p2_mask_gray = process_mask_for_pass(crop_mask_bgr, 2)
    p2_mask_pil = Image.fromarray(p2_mask_gray).resize((512, 512), resample=Image.Resampling.NEAREST)
    p2_prompt = selected_prompt + ", seamless hair edges, perfect boundary blending"
    p2_negative = base_negative + ", strange hair structure, extra curls, random loops, stray hair clumps, visible masking seams"
    output_pass2 = pipe(prompt=p2_prompt, negative_prompt=p2_negative, image=output_pass1, mask_image=p2_mask_pil,
                        num_inference_steps=35, guidance_scale=9.0, strength=0.40).images[0]
    res2_np = merge_step(output_pass2, res1_np, coords, p2_mask_gray)

    p3_mask_gray = process_mask_for_pass(crop_mask_bgr, 3)
    p3_mask_pil = Image.fromarray(p3_mask_gray).resize((512, 512), resample=Image.Resampling.NEAREST)
    p3_prompt = selected_prompt + ", flawless seamless hair boundaries, clean pixel blending"
    p3_negative = p2_negative + ", edge noise, pixel jitter"
    output_pass3 = pipe(prompt=p3_prompt, negative_prompt=p3_negative, image=output_pass2, mask_image=p3_mask_pil,
                        num_inference_steps=20, guidance_scale=8.5, strength=0.15).images[0]
    res3_np = merge_step(output_pass3, res2_np, coords, p3_mask_gray)

    return Image.fromarray(res1_np), Image.fromarray(res2_np), Image.fromarray(res3_np)



def update_styles_dropdown(gender, length):
    styles = list(HAIRSTYLES_DB[gender][length].keys()) + ["Свой вариант"]
    return gr.update(choices=styles, value=styles[0])


def toggle_custom_prompt_visibility(style_name):
    if style_name == "Свой вариант":
        return gr.update(visible=True, value="")
    return gr.update(visible=False, value="")


with gr.Blocks(title="ИИ Подбор Причесок") as demo:
    gr.Markdown("Виртуальный ИИ-Стилист")
    gr.Markdown("[GitHub с кодом](https://github.com/cfvvde/AI_Hair_Style)")

    with gr.Row():
        with gr.Column(scale=1):
            input_img_ui = gr.Image(label="Загрузить фото лица", type="filepath")

            gender_ui = gr.Radio(choices=["Мужская", "Женская"], value="Мужская", label="Тип прически")
            length_ui = gr.Radio(choices=["Короткие", "Длинные"], value="Короткие", label="Длина волос")

            init_styles = list(HAIRSTYLES_DB["Мужская"]["Короткие"].keys()) + ["Свой вариант"]
            style_ui = gr.Dropdown(choices=init_styles, value=init_styles[0], label="Доступные модели стрижек")

            # Текстовое поле ввода для своего варианта (скрыто по умолчанию)
            custom_prompt_ui = gr.Textbox(
                label="Опишите желаемую прическу (на английском)",
                placeholder="Например: dark blue messy undercut, cyberpunk style, k-pop haircut",
                visible=False,
                lines=2
            )

            # Логика интерактивного изменения UI элементов
            gender_ui.change(update_styles_dropdown, inputs=[gender_ui, length_ui], outputs=style_ui)
            length_ui.change(update_styles_dropdown, inputs=[gender_ui, length_ui], outputs=style_ui)
            style_ui.change(toggle_custom_prompt_visibility, inputs=[style_ui], outputs=custom_prompt_ui)

            submit_btn = gr.Button("Сгенерировать новый стиль", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Результаты обработки:")
            with gr.Tabs():
                with gr.TabItem("Финальный результат (Пасс 3)"):
                    output_res3 = gr.Image(label="Итоговый кадр с полировкой (strength=0.15)")
                with gr.TabItem("Очистка контуров (Пасс 2)"):
                    output_res2 = gr.Image(label="Сглаживание краев без артефактов (strength=0.40)")
                with gr.TabItem("Базовая геометрия (Пасс 1)"):
                    output_res1 = gr.Image(label="Первичная ИИ структура (strength=0.95)")

    submit_btn.click(
        fn=generate_web_interface,
        inputs=[input_img_ui, gender_ui, length_ui, style_ui, custom_prompt_ui],
        outputs=[output_res1, output_res2, output_res3]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
