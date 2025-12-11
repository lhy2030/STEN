環境：Google Colab<br>
資料：將 https://drive.google.com/drive/folders/1o1QtIsXUnIg4jwUHePUOeSwc_AGRoHIN 上傳至自己的雲端硬碟<br>
執行：<br>
!git clone https://github.com/lhy2030/STEN.git<br>
%cd STEN<br>
!pip install torch torchvision torchaudio<br>
!pip install numpy scipy scikit-learn pandas tqdm<br>
!pip install optuna<br>
from google.colab import drive<br>
drive.mount('/content/drive')<br>
!ln -s /content/drive/MyDrive/STEN_dataset ./data<br>
!python main.py --model STEN --runs 1 --data Epilepsy
