clear
close all

cd '/Users/stephane/Desktop/Canu/CAP_2018/competition'

%%

opts = detectImportOptions('data/train_cap2018.csv');
T = readtable('data/train_cap2018.csv',opts);
opts = detectImportOptions('data/train_cap2018.csv');
T2 = readtable('data/test_cap2018.csv',opts);

%%

X = table2array(T(:,[2:51,54:end-1]));
[n,p] = size(X);
X = (X - ones(n,1)*mean(X))./(ones(n,1)*std(X));
yl = table2cell(T(:,end));

[C, ia, y]  = unique(yl);

%ind_asaquer =  find(isnan(median(X)));
%X(:,ind_asaquer) = [];

%%

addpath(genpath('libsvm-3.22/'));


options='-c 100 -g 0.01 -m 200 -q';
ind = cvpartition(y,'Holdout',1/3);

 tic
 svm_model=svmtrain(y(ind.training), X(ind.training,:), options);
 [y_predict,accuracy,prob_estimates]=svmpredict(y(ind.test), X(ind.test,:), svm_model);
 fprintf('Time: %0.2f minutes. \n', toc/60);

%%

 C = Eval_CAp_2018(y_predict,y(ind.test))
 
 %   0.4471
 
%%
return


function C = Eval_CAp_2018(y_estimated,y_real)

% M is a 6 by 6 confision matrix real/prediction
% y_real = load(test_y_cap2018)

M = confusionmat(y_real,y_estimated)

Cost_M =[     0     1     2     3     4     6
     1     0     1     4     5     8
     3     2     0     3     5     8
    10     7     5     0     2     7
    20    16    12     4     0     8
    44    38    32    19    13     0];

C = 100*sum(sum(M.*Cost_M))/sum(sum(M));

end

% il faut verifier q'un même fichier donne les même résultats en matlab,
% python et R
% bb = [y_predict y(ind.test)];
% save('toto.txt','bb','-ascii')







