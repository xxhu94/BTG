BTG
====

PyTorch implementation for the paper below:  
**BTG: A Bridge to Graph machine learning in telecommunications fraud detection**



## Running BTG
To run the code, you need to have at least Python 3.7 or later versions. And follow the steps below :  
1.Go to [this site](https://aistudio.baidu.com/aistudio/datasetdetail/40690) to download the 4 datasets, namely train_app.csv,train_sms.csv,train_user.csv,train_voc.csv;  
2.Put the 4 downloaded datasets in the path: /BTG/data/train;  
3.Run `python data_process.py` to extract features and generate dataset in DGL;  
4.Run `python train.py ` to run BTG with default settings on the dataset.  

## Repo Structure
The repository is organized as follows:
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`:  training and testing BTG;
- `model.py`: BTG model implementations;
- `layers.py`: model layers;
- `utils.py`: utility functions;  
- `data`: raw dataset in /data/train, and extracted dataset in /data/user_data.  
 
## Citation

```
@article{hu2022btg,
  title={BTG: A Bridge to Graph machine learning in telecommunications fraud detection},
  author={Hu, Xinxin and Chen, Hongchang and Liu, Shuxin and Jiang, Haocong and Chu, Guanghan and Li, Ran},
  journal={Future Generation Computer Systems},
  volume={137},
  pages={274--287},
  year={2022},
  publisher={Elsevier}
}
```