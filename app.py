from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = ["bezhevyi", "belyi", "biryuzovyi", "bordovyi", "goluboi", "zheltyi", "zelenyi", "zolotoi", "korichnevyi",
          "krasnyi", "oranzhevyi", "raznocvetnyi", "rozovyi", "serebristyi", "seryi", "sinii", "fioletovyi", "chernyi"]

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(COLORS), drop_rate=0.3)
model = model.to(device)

checkpoint = torch.load("general_3_reg.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def detect_color(image_path):
    """
    Загружает фото товара, применяет необходимые преобразования
    и возвращает 5 лучших прогнозируемых цветов для него и еще возвращает
    18 лучших Распознанных цветов с вероятностями .
    """
     

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)

        top5_vals, top5_idx = torch.topk(probabilities, 5)
        color_probabilities_5 = {
            COLORS[idx.item()]: format(top5_vals[0][i].item(), ".4f") for i, idx in enumerate(top5_idx[0])
        }

    return color_probabilities_5  


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def model_page():
    color_probabilities_19 = {}
    color_probabilities_5 = {}
    color=""

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            color_probabilities_5 = detect_color(save_path)
            color= list(color_probabilities_5.keys())[0]

    return render_template('model.html', color=color, color5=color_probabilities_5)


if __name__ == '__main__':
    app.run(debug=True)
