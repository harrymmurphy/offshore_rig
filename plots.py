import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

# List of file paths to the forecast images
forecast_images = [
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Low_Spec_forecast.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Jackup_forecast.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Harsh_Semi_forecast.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\5G_6G_forecast.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\8G_7G_forecast.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Semi_Low_Spec_forecast.png'
]

# List of file paths to the validation images
validation_images = [
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Low_Spec_validation.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Jackup_validation.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Harsh_Semi_validation.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\5G_6G_validation.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\8G_7G_validation.png',
    r'C:\Users\Harry Murphy\OneDrive\Desktop\Semi_Low_Spec_validation.png'
]

# Titles for the images
titles = [
    "Low Spec Drillships",
    "Jackups",
    "Harsh Environment Semisubs",
    "5G/6G Drillships",
    "7G/8G Drillships",
    "Benign Environment Semisubs"
]

# Function to combine images into a grid with titles
def combine_images_with_titles(image_paths, titles, output_path, rows=3, cols=2):
    images = [Image.open(img) for img in image_paths]
    
    # Load a font
    font = ImageFont.load_default()

    # Calculate the size of each title bar
    title_height = 40  # Adjust the height of the title area

    # Get the size of the images
    widths, heights = zip(*(img.size for img in images))
    
    # Calculate the size of the combined image
    total_width = cols * max(widths)
    total_height = rows * (max(heights) + title_height)
    
    # Create a blank canvas
    combined_image = Image.new('RGB', (total_width, total_height), color='white')
    
    # Draw images and titles on the canvas
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Draw the title above the image
        title_area = Image.new('RGB', (img.width, title_height), color='white')
        title_draw = ImageDraw.Draw(title_area)
        title_draw.text((10, 10), titles[i], font=font, fill='black')
        
        # Combine the title and image
        combined_image.paste(title_area, (col * max(widths), row * (max(heights) + title_height)))
        combined_image.paste(img, (col * max(widths), row * (max(heights) + title_height) + title_height))
    
    # Save the combined image
    combined_image.save(output_path)

# Combine forecast images with titles
combine_images_with_titles(forecast_images, titles, r'C:\Users\Harry Murphy\OneDrive\Desktop\combined_forecast.png', rows=3, cols=2)

# Combine validation images with titles
combine_images_with_titles(validation_images, titles, r'C:\Users\Harry Murphy\OneDrive\Desktop\combined_validation.png', rows=3, cols=2)
