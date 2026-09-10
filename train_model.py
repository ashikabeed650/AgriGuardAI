from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

SEED = 42
IMAGE_SIZE = (224, 224)

# Add new vegetables here as soon as you have a trained dataset for them.
# The project will automatically derive the class list and folder mapping from this config.
CROP_DEFINITIONS = {
    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    ],
    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___healthy",
    ],
    "Pepper": [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
    ],
}

CLASS_NAMES = [label for crop_labels in CROP_DEFINITIONS.values() for label in crop_labels]
DATASET_FOLDERS = {
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___healthy": "Tomato_healthy",
    "Pepper__bell___Bacterial_spot": "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy": "Pepper__bell___healthy",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_files(dataset_root: Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / DATASET_FOLDERS[class_name]
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        class_files = sorted(
            str(path) for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_files:
            raise ValueError(f"No images found in {class_dir}")
        paths.extend(class_files)
        labels.extend([label_index] * len(class_files))
        print(f"{class_name}: {len(class_files)} images")
    return paths, labels


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, len(CLASS_NAMES))


def make_dataset(paths: list[str], labels: list[int], batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ])
        dataset = dataset.map(lambda image, label: (augmentation(image, training=True), label), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def stratified_split(paths: list[str], labels: list[int], validation_fraction: float = 0.2) -> tuple[list[str], list[str], list[int], list[int]]:
    rng = np.random.default_rng(SEED)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    labels_array = np.asarray(labels)
    for label_index in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels_array == label_index)
        rng.shuffle(class_indices)
        validation_count = max(1, int(round(len(class_indices) * validation_fraction)))
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return (
        [paths[index] for index in train_indices],
        [paths[index] for index in validation_indices],
        [labels[index] for index in train_indices],
        [labels[index] for index in validation_indices],
    )


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AgriGuard's 9-class leaf disease model.")
    parser.add_argument("--dataset", type=Path, default=Path("datsets/PlantVillage"))
    parser.add_argument("--base-model", type=Path, default=Path("model/agriguard_model.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/trained"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    paths, labels = collect_files(args.dataset)
    train_paths, validation_paths, train_labels, validation_labels = stratified_split(paths, labels)
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(validation_paths)}")

    train_dataset = make_dataset(train_paths, train_labels, args.batch_size, training=True)
    validation_dataset = make_dataset(validation_paths, validation_labels, args.batch_size, training=False)

    model = tf.keras.models.load_model(args.base_model, compile=False)
    if model.output_shape[-1] != len(CLASS_NAMES):
        print(f"Base model has {model.output_shape[-1]} outputs; replacing final classification head with {len(CLASS_NAMES)} classes.")
        inputs = model.input
        x = model.layers[-2].output
        outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="new_classification_head")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model.name)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-3:]:
        layer.trainable = True

    label_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    class_weights_array = len(train_labels) / (len(CLASS_NAMES) * label_counts)
    class_weights = {index: float(min(weight, 4.0)) for index, weight in enumerate(class_weights_array)}
    print("Class weights:", json.dumps(dict(zip(CLASS_NAMES, class_weights_array.round(2).tolist()))))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    metrics = model.evaluate(validation_dataset, return_dict=True)
    print("Final validation metrics:", json.dumps({key: float(value) for key, value in metrics.items()}))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras_output = args.output_dir / "agriguard_model.keras"
    tflite_output = args.output_dir / "agriguard_model.tflite"
    model.save(keras_output)
    export_tflite(model, tflite_output)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps({"classes": CLASS_NAMES, "metrics": {key: float(value) for key, value in metrics.items()}, "history": history.history}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {keras_output}")
    print(f"Saved: {tflite_output}")


if __name__ == "__main__":
    main()
