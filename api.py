from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import uuid

# Define the Model Architecture (Generator)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU()
        )
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        bottleneck = self.bottleneck(d6)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        return self.final_up(torch.cat([u6, d1], 1))

# Initialize FastAPI App
app = FastAPI()

# Configuration for output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)
app.mount("/generated_images", StaticFiles(directory=output_dir), name="generated_images")

device = "cpu"
model_path = "gen.pth.tar"

# Initialize and Load Model
model = Generator(in_channels=1).to(device)

try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Success: Model weights loaded.")
except Exception as e:
    print(f"Error: Could not load model. Details: {e}")

# API Endpoint
@app.post("/predict")
async def predict_mri(file: UploadFile = File(...), request: Request = None):
    try:
        # 1. Read the uploaded file and resize
        file_content = await file.read()
        original_image = Image.open(io.BytesIO(file_content)).convert("L")
        resized_image = original_image.resize((256, 256), Image.BICUBIC)
        image_array = np.array(resized_image)
        image_array = (image_array / 127.5) - 1.0 
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # 2. Inference
        with torch.no_grad():
            generated_tensor = model(input_tensor)
        
        # 3. Post-processing 
        output_array = generated_tensor.squeeze().cpu().numpy()
        output_array = (output_array + 1) / 2.0 * 255.0 
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        new_image = Image.fromarray(output_array)
        
        # 4. Save Image 
        unique_filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join(output_dir, unique_filename)
        new_image.save(save_path)
        
        # 5. Generate URL 
        base_url = str(request.base_url) 
        image_url = f"{base_url}{output_dir}/{unique_filename}"
        
        return {
            "status": "success",
            "image_url": image_url
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)