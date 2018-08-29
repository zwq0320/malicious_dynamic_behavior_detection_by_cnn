# 使用CNN进行样本恶意动态行为检测  

## 运行方法  
1.  格式转换json->txt
    1.  将原始cuckoo跑出的动态行为报告report.json都拷贝到文件夹data/cuckoo_reports中 
    2.  用下面指令转换cuckoo报告格式,结果保存在data/cuckoo_report_txts中  
```
python cuckoo2txt.py
```   

2.  准备训练样本  
    把转换为txt格式的动态行为报告，根据类别存放到data/train下的不同目录。  
    例：二分类，分别把样本存入data/train/pos和data/train/neg   
    
3.  训练  
    main函数中有可调节的参数，可根据需要修改。
```
python train.py
```   

4.  测试  
    将txt格式的测试文件存入data/test子目录下   
```
python test.py
```    

5. 查看tensorboard   
```
tensorboard --logdir runs/
```