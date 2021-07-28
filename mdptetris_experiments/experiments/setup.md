# Experiment setup commands
## To download and set up repos 
```bash
git clone https://github.com/Benjscho/gym-mdptetris
cd ./gym-mdptetris && pip3 install -e .
cd ..
git clone https://github.com/Benjscho/mdptetris-experiments
cd ./mdptetris-experiments && pip3 install -e .
```

## Setup virtual env

```bash
python3 -m venv --system-site-packages ./tetris-venv
```
## Get correct version of torch for cuda 11.0 (on garlick)
`pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`


## Port forwarding for tensorboard
```bash
ssh -L 8080:localhost:6006 bjs82@garlick.cs.bath.ac.uk 
ssh -L <local machine port>:localhost:<remote machine port> <username>@example.com
```

## See reserved disk 
`hare me`

## Retrieve files
```
# Compress runs 
tar -cvzf runs.tar.gz ./runs 
# Retrieve them with scp
scp bjs82@garlick.cs.bath.ac.uk:/mnt/fast0/bjs82/runs.tar.gz /Users/crow/Desktop/bath-msc-main/cm50170-Diss/experiment-results
# Extract 
tar -xvzf runs.tar.gz
```
