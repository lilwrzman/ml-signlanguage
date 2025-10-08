import os
import shutil

from pathlib import Path
from ultralytics import YOLO

class YoloTrainer:
    def __init__(self,
                 model_name="yolo11x.pt",
                 data_yaml="config/data.yaml",
                 epochs=50,
                 batch=16,
                 imgsz=640):
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.output_dir = "runs/detect/train"
        
    def train_model(self):
        """Train YOLO model"""
        print("🚀 Starting YOLO11x training...")
        print(f"📦 Model: {self.model_name}")
        print(f"📄 Data:  {self.data_yaml}")
        print(f"⚙️ Epochs: {self.epochs}, Batch: {self.batch}, Image Size: {self.imgsz}\n")
        
        target_path = Path("models/checkpoints/yolo11x.pt")
        if target_path.exists():
            print(f"📦 Default YOLO model already exists at: {target_path}")
            model = YOLO(str(target_path))
        else:
            model = YOLO("yolo11x.pt")
            default_path = Path(model.ckpt_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(default_path, target_path)
            print("📦 Default YOLO model available on:", target_path)
            
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=0,
            workers=4,
            patience=10,
            amp=True,
            cos_lr=True,
            cache="disk",
            project="runs/detect",
            name="train",
            exist_ok=True,
        )
        
        print("✅ Training complete!")
        print(f"📂 Results saved in: {results.save_dir}")
        
        return results
    
    def evaluate_model(self, weights_path=None):
        """Evaluate trained model performance"""
        weights_path = weights_path or os.path.join(self.output_dir, "weights", "best.pt")

        print("\n🔍 Evaluating model performance...")
        model = YOLO(weights_path)
        metrics = model.val(data=self.data_yaml)

        print("✅ Evaluation complete!")
        print(f"📊 Results: {metrics.results_dict}")
        return metrics
    
    def run(self):
        """Run full training + evaluation pipeline"""
        results = self.train_model()
        self.evaluate_model(weights_path=os.path.join(results.save_dir, "weights", "best.pt"))
        print("\n🏁 Training pipeline finished successfully!")
        
if __name__ == "__main__":
    trainer = YoloTrainer()
    trainer.run()
        