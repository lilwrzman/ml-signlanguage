import os
import shutil
import zipfile

from glob import glob
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self, data_dir="data/raw", processed_dir="data/processed", dataset_id="grassknoted/asl-alphabet", zip_name="asl-alphabet.zip"):
        self.root_dir = "."
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.dataset_id = dataset_id
        self.zip_name = zip_name
        self.extract_dir = None
        
        load_dotenv()
        
        
    def setup_kaggle_credentials(self):
        """Setup Kaggle credentials from .env file"""
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            raise ValueError("❌ Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

        if not os.path.exists(kaggle_json):
            print("🔑 Writing Kaggle credentials...")
            with open(kaggle_json, "w") as f:
                f.write(f'{{"username":"{username}","key":"{key}"}}')
            os.chmod(kaggle_json, 0o600)
        else:
            print("✅ Kaggle credentials already exist.")
            
            
    def download_dataset(self):
        """Download dataset from Kaggle if not already present"""
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if os.path.exists(zip_path):
            print(f"📦 Dataset ZIP already exists: {zip_path}")
            return

        print("⬇️ Downloading dataset from Kaggle...")
        os.system(f'kaggle datasets download -d {self.dataset_id} -p {self.data_dir}')
        print("✅ Download complete.")
        
        
    def extract_dataset(self):
        """Extract dataset if not already extracted; flatten nested folders if needed."""
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if not os.path.exists(zip_path):
            raise FileNotFoundError("❌ Dataset ZIP not found. Run download_dataset() first.")

        # Check if already extracted
        existing_folders = [
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
            and f.lower().startswith("asl")
        ]

        if any("train" in f.lower() for f in existing_folders) and any("test" in f.lower() for f in existing_folders):
            print("🗂️ Dataset already extracted (train & test folders found). Skipping extraction.")
        else:
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("✅ Extraction complete.")

        # Detect and flatten nested folder (e.g., asl_alphabet_train/asl_alphabet_train)
        for root, dirs, _ in os.walk(self.data_dir):
            for d in dirs:
                inner = os.path.join(root, d, d)
                outer = os.path.join(root, d)
                if os.path.exists(inner):
                    print(f"⚙️ Flattening nested folder: {inner}")
                    for item in os.listdir(inner):
                        src = os.path.join(inner, item)
                        dst = os.path.join(outer, item)
                        os.rename(src, dst)
                    os.rmdir(inner)
                    print(f"✅ Folder flattened: {outer}")

        # Auto-detect extracted folders again (post-flatten)
        extracted_folders = [
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
            and f.lower().startswith("asl")
        ]

        if not extracted_folders:
            raise FileNotFoundError("❌ Could not find any extracted ASL folder in data/raw/")

        # Prefer train folder for labeling
        train_folder = [f for f in extracted_folders if "train" in f.lower()]
        if train_folder:
            self.extract_dir = os.path.join(self.data_dir, train_folder[0])
        else:
            self.extract_dir = os.path.join(self.data_dir, extracted_folders[0])

        print(f"📁 Using extracted dataset folder: {self.extract_dir}")
        
        
    def generate_labels(self):
        """Generate YOLO labels (1 class per folder, full-image bounding boxes)"""
        classes = sorted(os.listdir(self.extract_dir))
        image_dir = os.path.join(self.processed_dir, "images")
        label_dir = os.path.join(self.processed_dir, "labels")
        names_path = os.path.join(self.processed_dir, "classes.txt")
        
        if (
            os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0
            and os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0
            and os.path.exists(names_path)
        ):
            print("🗂️ Detected existing labeled dataset. Skipping labeling process.")
            self.classes = open(names_path).read().splitlines()
            print(f"📄 Loaded {len(self.classes)} existing class names.")
            return

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"🏷️ Generating labels for {len(classes)} classes...")
        for idx, cls in enumerate(classes):
            class_path = os.path.join(self.extract_dir, cls)
            if not os.path.isdir(class_path):
                continue
            
            img_paths = glob(os.path.join(class_path, "*.jpg"))
            print(f"🔠 Processing class [{idx}] '{cls}' → {len(img_paths)} images")
            
            for i, img_path in enumerate(img_paths):
                filename = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(image_dir, filename))
                label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write(f"{idx} 0.5 0.5 1.0 1.0\n")
                    
                if (i + 1) % 1000 == 0:
                    print(f"  ↳ {i+1} files processed for class '{cls}'")

        print("✅ Label generation complete.")

        # Write class names
        names_path = os.path.join(self.processed_dir, "classes.txt")
        with open(names_path, "w") as f:
            f.write("\n".join(classes))
        print("📄 Class list saved at:", names_path)
        
        
    def split_dataset(self):
        """Split dataset into train/val/test sets"""
        split_root = os.path.join(self.processed_dir, "images", "train")
        val_root = os.path.join(self.processed_dir, "images", "val")
        test_root = os.path.join(self.processed_dir, "images", "test")
        
        if (
            os.path.exists(split_root) and len(os.listdir(split_root)) > 0
            and os.path.exists(val_root) and len(os.listdir(val_root)) > 0
            and os.path.exists(test_root) and len(os.listdir(test_root)) > 0
        ):
            print("🗂️ Detected existing dataset split. Skipping split process.")
            print(f"📊 Existing split summary:")
            print(f"   Train: {len(os.listdir(split_root))} images")
            print(f"   Val:   {len(os.listdir(val_root))} images")
            print(f"   Test:  {len(os.listdir(test_root))} images")
            return
        
        images = glob(os.path.join(self.processed_dir, "images", "*.jpg"))
        train, test = train_test_split(images, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)

        for subset, files in zip(["train", "val", "test"], [train, val, test]):
            img_dir = os.path.join(self.processed_dir, "images", subset)
            lbl_dir = os.path.join(self.processed_dir, "labels", subset)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            
            print(f"📂 Splitting subset: {subset.upper()} ({len(files)} images)")

            for i, file in enumerate(files):
                shutil.copy(file, img_dir)
                lbl_file = file.replace("images", "labels").replace(".jpg", ".txt")
                shutil.copy(lbl_file, lbl_dir)
                
                if (i + 1) % 1000 == 0 or (i + 1) == len(files):
                    print(f"  ↳ {i+1}/{len(files)} files copied to {subset}")
                    
        print("✅ Dataset split complete.")
        print(f"📊 Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        
        
    def create_data_config(self):
        """Generate YOLO data.yaml automatically based on processed dataset"""
        config_dir = os.path.join(self.root_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        yaml_path = os.path.join(config_dir, "data.yaml")

        classes_path = os.path.join(self.processed_dir, "classes.txt")
        if not os.path.exists(classes_path):
            raise FileNotFoundError("❌ classes.txt not found. Run generate_labels() first.")

        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

        yaml_content = f"""train: ../data/processed/images/train
val: ../data/processed/images/val
test: ../data/processed/images/test

nc: {len(class_names)}
names: {class_names}
    """

        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print("📝 YOLO dataset configuration file created:")
        print(f"📄 Path: {yaml_path}")
        print(f"📊 Classes ({len(class_names)}): {class_names}")
        
        
    def prepare_dataset(self):
        """Full pipeline: credentials → download → extract → label → split"""
        print("🚀 Starting ASL dataset preparation pipeline...")
        self.setup_kaggle_credentials()
        self.download_dataset()
        self.extract_dataset()
        self.generate_labels()
        self.split_dataset()
        self.create_data_config()
        print("🎉 Dataset preparation complete!")

        
if __name__ == "__main__":
    handler = DatasetHandler()
    handler.prepare_dataset()