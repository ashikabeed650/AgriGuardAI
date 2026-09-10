# AgriGuard crop expansion plan

The current trained model supports Tomato, Potato, and Pepper. New crops should only be added after both training images and validation images are available.

## Requested crops and recommended class scope

| Crop family | Suggested classes                                                        | Dataset search/source                                                      |
| ----------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| Rice        | healthy, bacterial leaf blight, brown spot, leaf smut                    | Kaggle: `Rice Leaf Diseases`; Mendeley Data rice leaf disease datasets     |
| Wheat       | healthy, brown rust, yellow rust, septoria, mildew                       | Kaggle: `Wheat Leaf Disease`; PlantDoc where wheat labels are available    |
| Jowar       | healthy, rust, leaf blight                                               | Search `sorghum/jowar leaf disease dataset` on Kaggle or Mendeley Data     |
| Bajra       | healthy, downy mildew, rust                                              | Search `pearl millet/bajra disease dataset` on Kaggle or Mendeley Data     |
| Ragi        | healthy, blast, leaf spot                                                | Search `finger millet/ragi disease dataset` on Kaggle or Mendeley Data     |
| Maize/Corn  | healthy, common rust, northern leaf blight, gray leaf spot               | Official PlantVillage mirror: `mohanty/PlantVillage` on Hugging Face       |
| Pulses      | Choose a specific crop first: chickpea, pigeon pea, mung bean, or lentil | Search the chosen crop plus `leaf disease dataset` on Kaggle/Mendeley Data |
| Sugarcane   | healthy, red rot, mosaic, rust, yellow leaf disease                      | Kaggle: `Sugarcane Leaf Disease Dataset`                                   |
| Cotton      | healthy, bacterial blight, curl virus, fusarium wilt                     | Kaggle: `Cotton Plant Disease Dataset`                                     |
| Oilseeds    | Choose a specific crop first: groundnut, mustard, soybean, or sunflower  | Search the chosen crop plus `leaf disease dataset` on Kaggle/Mendeley Data |

## Required folder format

Place the downloaded images under `datsets/PlantVillage` using one folder per exact class label, for example:

```text
Rice___healthy/
Rice___Bacterial_leaf_blight/
Rice___Brown_spot/
Rice___Leaf_smut/
```

Each crop must have at least one healthy class and disease classes with enough images for a stratified validation split. Do not mix labels from different datasets until their class names and image quality have been checked.

After the folders are present, run:

```powershell
venv\Scripts\python.exe train_model.py --base-model model\agriguard_model.keras --output-dir model\trained --epochs 8 --batch-size 32
```

Then add the same crop/class definitions to `app.py` and `index.html`, copy the validated TFLite file to `model\agriguard_model.tflite`, and restart Flask.
