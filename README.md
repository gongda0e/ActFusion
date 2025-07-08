# ActFusion: A Unified Diffusion Model for Action Segmentation and Anticipation (NeurIPS 2024)

📄[ Paper (arXiv)](https://arxiv.org/abs/2412.04353)

This repository provides the official implementation of our NeurIPS 2024 paper:

> **ActFusion: A Unified Diffusion Model for Action Segmentation and Anticipation** <br>
> [Dayoung Gong](https://gongda0e.github.io/), Suha Kwak, and Minsu Cho <br>
> *NeurIPS, Vancouver, 2024*

<p align="center">
  <img src="assets/pipeline.png" alt="An Overview of ActFusion" />
</p>

---

## 🛠️ Recommended Environment & Installation

**Recommended Environment**  
- Python 3.8.20  
- CUDA 11.7  
- PyTorch 1.13.0+cu117  

**Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Setup

Download the preprocessed dataset from [this link](https://mega.nz/file/O6wXlSTS#wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) (borrowed from [MS-TCN](https://github.com/yabufarha/ms-tcn)).

Create a directory structure as below, and place the datasets inside the `datasets/` folder:

```
project-root/
├── ckpt/                 # pretrained model checkpoints
│   ├── breakfast/
│   └── 50salads/
├── configs/              # auto-generated JSON config files
│   ├── Breakfast.json
│   └── 50salads.json
├── datasets/             # downloaded datasets
│   ├── breakfast/
│   └── 50salads/
├── result/               # experiment outputs will be saved here
├── src/                  # source code
│   ├── model/
│   │   ├── actfusion.py
│   │   ├── backbone.py
│   │   ├── attn.py
│   │   └── __init__.py
│   ├── dataset.py
│   ├── default_configs.py
│   ├── trainer.py
│   ├── utils.py
│   ├── vis.py
│   └── __init__.py
├── main.py
├── LICENSE
└── README.md
```

---

## 🚀 Training

Generate config files by running:

```bash
python default_configs.py
```

Then start training with:

```bash
python main.py --config configs/Breakfast.json --result_dir $result_dir --split $split_num
```

---

## 🧪 Testing with Pretrained Checkpoints

1. Download pretrained checkpoints from [this link](https://drive.google.com/drive/u/1/folders/1Rnb4ZcpujFsQKeBpi4ZhZROc6RDSRlsg)
2. Place the downloaded folders inside the `ckpt/` directory
3. Run evaluation:

```bash
python main.py --config configs/Breakfast.json --result_dir $result_dir --split $split_num --test --ckpt
```

---

## 🙏 Acknowledgement & 📚 Citation

This repository builds upon the [DiffAct](https://github.com/Finspire13/DiffAct) codebase. We thank the original authors for sharing their work.

If you find our code or paper helpful, please consider citing both ActFusion and DiffAct:

```bibtex
@article{gong2024actfusion,
  title={ActFusion: A Unified Diffusion Model for Action Segmentation and Anticipation},
  author={Gong, Dayoung and Kwak, Suha and Cho, Minsu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={89913--89942},
  year={2024}
}

@inproceedings{liu2023diffusion,
  title={Diffusion Action Segmentation},
  author={Liu, Daochang and Li, Qiyue and Dinh, Anh-Dung and Jiang, Tingting and Shah, Mubarak and Xu, Chang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
