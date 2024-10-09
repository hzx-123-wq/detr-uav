1.If you want to run this code, you first need to configure the environment and run the following code：  pip install -r requirements.txt

2.train.py is a running program that can test the performance of the network without distillation. If you want to run it, you need to change the path of the dataset.

3.The teacher model in distill.py first trains the best weight file through train.py, and then correctly fills in the addresses of the student model and the teacher model, as well as the weight address of the teacher model. After the addresses are correctly classified, running distill.py can reproduce our results.

4.To reproduce our experimental results, you need to run the trained weight file in val.py. If you want to get data closer to the paper, you only need to change the path of the validation set and run it according to the parameters we have set.

5.The following are the files we have trained. You can directly test the performance of our model or view the results of our trained model：

  the best train of visdrone is on https://pan.baidu.com/s/1Fb8PiazIIsTwKtu-0wCoYQ?pwd=jy3r
	
  the best val of visdrone is on https://pan.baidu.com/s/1_dbkO7bf09bEHM92BIHOQg?pwd=v585
	
  the teacher of visdrone is on https://pan.baidu.com/s/1Rhye0FTMYcBf75XPh52QUw?pwd=pqd7
	
  the best train of tinyperson is on https://pan.baidu.com/s/1uQPyEAlyGJENxPeiPrMo8w?pwd=bmaq
	
  the best val of tinyperson is on https://pan.baidu.com/s/1lC0dAFebqSBpJQAcbEqJEQ?pwd=bn1r
	
  the teacher of tinyperson is on https://pan.baidu.com/s/1k2XXuvsluClxLpME3pbdbw?pwd=6jfq
	
  the per-train is on https://pan.baidu.com/s/1-uOAXIGOfXHwpfRlNPDIvg?pwd=skdx
