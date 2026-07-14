# bitte eine Enviroment erstellen und aktivieren 
# GPU-Pytorch-version installieren command:
# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# ultralytics installieren command: pip install ultralytics


from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data=r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\Fahrsimulator\System-Autonomes-Fahren--T-\Behavioural_Cloning_Basic\schilder\yolo\data_290_cls.yaml",
    epochs=150,
    imgsz=640,
    batch=4,
    device=0,

    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.6,
    blur=0.01,
    fliplr=0.5,
    mosaic=0.5,

    patience=100,
    save=True,
    plots=True,
    val=True
)
