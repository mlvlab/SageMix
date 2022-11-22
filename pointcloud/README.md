# SageMix for point cloud classification
## Installation
```
# Create and activate environment using conda
conda create -n SageMix -y python=3.7
conda activate SageMix
 
# Install pytorch (based on your CUDA version)
conda install -y pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
# Install requirements
pip install -r requirements.txt

# Install earth mover distance
cd emd_
python setup.py install
cd ..
```

## Data

**Notes** : When you run the `main.py`, dataset(ModelNet40, ScanObjectNN) is automatically downloaded at `.../SageMix/pointcloud/data/`.  
If you already have downloaded dataset on your `$PATH`, make a symbolic link at `.../SageMix/pointcloud/data/`. 

## Runnig the code

**Train**
```
CUDA_VISIBLE_DEVICES=$GPUs python main.py --exp_name=SageMix --model=$model --data=$data #--sigma=$sigma
```
- $GPUs is the list of GPUs to use
- $model is the name of model, currently we support {`dgcnn`, `pointnet`} for DGCNN and PointNet, respectively.
- $data is the name of dataset, currently we support {`MN40`, `SONN_easy`, `SONN_hard`} for ModelNet40, ScanObjectNN-OBJ_ONLY, and ScanObjectNN-PB_T50_RS, respectively.  
- $sigma is the bandwidth for kernel. Note that $sigma is automatically set to 0.3 for DGCNN and 2.0 for pointNet.

For instance, if you want to train DGCNN on ScanObjectNN-OBJ_ONLY using 2 GPU, see below.
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_name=SageMix --model=dgcnn --data=SONN_easy
```

**Test**
```
CUDA_VISIBLE_DEVICES=$GPUs python main.py --eval=True --exp_name=SageMix --model=$model --data=$data --model_path=$PATH
```
- Evaluation with trained model located at `$PATH`

## Contact
Let me know if you have any questions or concerns.
> Sanghyeok Lee, cat0626@korea.ac.kr

## Acknowledgement

We borrow the codebase from [DGCNN](https://github.com/WangYueFt/dgcnn).  
We use the code for EMD implemented in [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion).
