環境：Google Colab
資料：將https://drive.google.com/drive/folders/1o1QtIsXUnIg4jwUHePUOeSwc_AGRoHIN上傳至自己的雲端硬碟
執行：
!git clone https://github.com/lhy2030/STEN.git
%cd STEN

!pip install torch torchvision torchaudio
!pip install numpy scipy scikit-learn pandas tqdm
!pip install optuna

from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/STEN_dataset ./data

!python main.py --model STEN --runs 1 --data Epilepsy
