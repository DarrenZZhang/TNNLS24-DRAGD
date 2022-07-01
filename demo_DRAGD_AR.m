% implemented on Matlab2015a
% The code is written by Jie Wen, if you have any questions, please contact
% jiewen_pr@126.com
% Note: if you use the code, please cite:
% J Wen, S Deng, L Fei, et al., Discriminative Regression with Adaptive Graph Diffusion[J]. TNNLS, 2022.

clear all
clc

Dataname = 'AR_50x40_jerry';

sele_num = 4; 
lambda1 = 0.1;
lambda2 = 0.0001;
lambda3 = 0.01;
rand('seed',1847);

load (Dataname);
nnClass = length(unique(gnd));
num_Class = [];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))]; 
end

fea = double(fea);
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx = find(gnd==j);
    randIdx = randperm(num_Class(j));
    Train_Ma = [Train_Ma;fea(idx(randIdx(1:sele_num)),:)];           
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = NormalizeFea(Train_Ma',0);                      
Test_Ma  = NormalizeFea(Test_Ma',0);
label = unique(Train_Lab);

options = [];
options.k = 0;
options.WeightMode = 'HeatKernel';
Z = constructW(Train_Ma',options);        
Z = full(Z);
Z = (Z+Z')*0.5;

miu = 0.01;
rho = 1.2;
max_iter = 100;
max_iter_diffusion = 10; 

W = DRAGD(Train_Ma,Z,Train_Lab,lambda1,lambda2,lambda3,miu,rho,max_iter,max_iter_diffusion);

Train_Maa = W*Train_Ma;
Test_Maa  = W*Test_Ma;
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);    
[class_test] = knnclassify(Test_Maa', Train_Maa', Train_Lab,1,'euclidean','nearest');
rate_KNN = sum(Test_Lab == class_test)/length(Test_Lab)*100