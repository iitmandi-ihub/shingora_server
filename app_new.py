from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import os

# Define the class labels
class_labels = ['Low/Naps', 'Low/Slubs', 'Low/Thick_Thin', 'Medium/Naps',"Medium/Slubs","Medium/Thick-Thin","Pure","unknown"]
num_classes=8
# Define the model class
class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize Flask app
app_new = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Model(num_classes)
model.load_state_dict(torch.load('best_resnet50_model.pth', map_location=device))
model.to(device)
model.eval()

# Define the transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the function to predict a single image
def predict_image(image_path, class_labels):
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        predicted_label = class_labels[predicted_idx]
        defect_name, defect_type = parse_class_name(predicted_label)
    
    return predicted_idx, defect_name, defect_type


# Parse class name into category and defect
def parse_class_name(class_name):
    parts = class_name.split('/')
     #print(f"Print class name: {class_name}")
    if len(parts) == 2:
        category, defect = parts
    else:
        category, defect = parts[0], 'None'
    return category, defect
# Test Api
@app_new.route('/test',methods=['GET'])
def test():
    return jsonify({'message': 'This is for testing'}),200
# Create an API route for image prediction
@app_new.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image parameter found in the request'}), 400

    image_file = request.files['image']
    
    # Save the image to a temporary location
    image_path = './temp_image.jpg'
    image_file.save(image_path)

    try:

        # Get prediction for the image
        predicted_class_idx, defect_name, defect_type = predict_image(image_path, class_labels)
       

        response = {
            'success': True,
            'message': 'Prediction successful',
            'predicted_class_index': predicted_class_idx,
            'predicted_defect_name': defect_name,
            'predicted_defect_type':defect_type
        }

        return jsonify(response), 200 
    except Exception as e:
        return jsonify({
            'success': False,
            'message':f'{str(e)}',
        }), 500
    finally:
        # Clean up temporary image
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == '__main__':
    app_new.run(debug=True,host="0.0.0.0", port=5000)