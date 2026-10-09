# Class-Incremental-Learning-via-LoRA-based-Elastic-Ensemble-of-Experts
Official implementation of NeurIPS2026 paper "Class-Incremental Learning via LoRA-based Elastic Ensemble of Experts"

## ▶️ Usage

### **1. Create env and install requirements**

```bash
conda create -n eee python=3.10
conda activate eee
pip install -r requirements.txt (wait a moment)
```

### **2. Run the example training script**

```bash
bash eee.sh
```

### Project structure overview

```bash
eee/
├── backbone/                           # Pre-trained backbone models
│   ├── eee.py                          # eee backbone wrapper
│   ├── vision_transformer_timm.py      # eee backbone implementation
│   └── ...
├── datasets/                           # Dataset loaders
|   ├── init.py              
|   ├── seq_imagenet_r.py               # Modify task numbers                
│   └── ...
├── models/                             # CL Method implementations
│   └── eee.py                          # eee method implementation
├── utils/                              # Helper tools
|   ├── training.py                     # Training scripts                
│   └── ...
├── main.py                             # Main entry
├── eee.sh
└── README.md
```

------

## 📝 Citation

If you find this repository helpful, please click the ⭐Star and cite our paper (wait a moment):

------

## 🙏 Acknowledgement

Thanks for the awesome continual learning framework **[Mammoth](https://github.com/aimagelab/mammoth)**.
The LoRA implementation is derived from **[SPT](https://github.com/ziplab/SPT)**. 
