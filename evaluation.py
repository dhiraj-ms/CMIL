import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


image_size = (224, 224)
batch_size = 32

def evaluate_model(image_folder):
    model = load_model('vgg16_model.h5')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    df = pd.DataFrame({
        'filename': image_files
    })
    
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(os.path.abspath(image_folder), x))

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',  
        y_col=None,  
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False
    )

    filenames = test_generator.filenames

    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    results_df = pd.DataFrame({
        'name': [os.path.basename(f) for f in filenames],  
        'predicted_labels': predicted_labels
    })

    output_csv = 'evaluation.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"Predicted labels saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate images in a folder and output predictions to a CSV.")
    parser.add_argument("image_folder", help="Path to the folder containing images.")
    
    args = parser.parse_args()

    evaluate_model(args.image_folder)

