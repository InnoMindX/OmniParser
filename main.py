from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
from io import BytesIO
from PIL import Image
import torch
import os
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import uuid
import tempfile

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
som_model.to(device)

# Two choices for caption model: fine-tuned blip2 or florence2
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")


class ImageRequest(BaseModel):
    image_base64: str
    platform: str = 'web'


class ImageResponse(BaseModel):
    dino_labeled_img: str
    labeled_data: List[dict]


@app.post("/process_image", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Save the image to a temporary BytesIO buffer to use as a file-like object
        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as temp_file:
            temp_image_path = temp_file.name
            image.save(temp_image_path)

            # OCR and object detection configurations
            platform = request.platform
            if platform == 'pc':
                draw_bbox_config = {
                    'text_scale': 0.8,
                    'text_thickness': 2,
                    'text_padding': 2,
                    'thickness': 2,
                }
            elif platform == 'web':
                draw_bbox_config = {
                    'text_scale': 0.8,
                    'text_thickness': 2,
                    'text_padding': 3,
                    'thickness': 3,
                }
            elif platform == 'mobile':
                draw_bbox_config = {
                    'text_scale': 0.8,
                    'text_thickness': 2,
                    'text_padding': 3,
                    'thickness': 3,
                }
            BOX_THRESHOLD = 0.03

            # Perform OCR
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(temp_image_path, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9})
            text, ocr_bbox = ocr_bbox_rslt

            # Process image with SOM and caption models
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                temp_image_path,
                som_model,
                BOX_TRESHOLD=BOX_THRESHOLD,
                output_coord_in_ratio=False,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=0.1
            )

            # Encode the labeled image back to base64
            labeled_image = Image.open(BytesIO(base64.b64decode(dino_labeled_img)))
            buffered = BytesIO()
            labeled_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Return results
            labeled_data = [
                {
                    "content": parsed_content_list[i],
                    "coordinates": label_coordinates[str(i)].tolist()
                }
                for i in range(len(parsed_content_list))
            ]

            return ImageResponse(
                dino_labeled_img=encoded_image,
                labeled_data=labeled_data
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8099, reload=True)


if __name__ == "__main__":
    main()

# To run the server, use the command: uvicorn filename:app --reload
# For example: uvicorn main:app --reload if this code is saved in main.py
