# Attribute Extraction Module

This directory contains code for extracting clothing attributes from images using a trained deep learning model.

## Files

- `attr_extractor.py`: Main script for loading the trained model and extracting attributes from input images.
- `model.py`: Contains the definition of the `AttributeResNet` model architecture.
- `list_attr_cloth.txt`: Text file listing all attribute names (one per line, format may vary).

## Usage

1. **Prepare the Model and Attribute File**
   - Ensure you have a trained model weights file (e.g., `model.pth`).
   - Ensure `list_attr_cloth.txt` is present and matches the attributes used during training.

2. **Extract Attributes from an Image**

   Example usage in Python:
   ```python
   from attr_extractor import AttributeExtractor
   import torch

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   extractor = AttributeExtractor(
       model_path='model.pth',
       device=device,
       attr_file='list_attr_cloth.txt',
       threshold=0.2  # Adjust as needed
   )
   attributes = extractor.extract('path_to_image.jpg')
   print(attributes)
   ```

   - The output is a list of probabilities for each attribute.
   - You can modify the code to return attribute names above a certain threshold if desired.

3. **Customizing Threshold**
   - The `threshold` parameter controls the minimum probability for an attribute to be considered present.

## Notes

- The model expects images to be RGB and will resize them to 224x224 pixels.
- For more details, see the code in `attr_extractor.py`.
