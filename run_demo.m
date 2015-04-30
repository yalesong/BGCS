%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo script for Bayesian Compressed Sensing (BCS) and Bayesian 
%   Group-Sparse Compressed Sensing (BGCS) 
%
% Before running this demo script, please download the following two files
% and place both files under the directory ./data/:
%
%  (1) http://people.csail.mit.edu/yalesong/fg15/CK+_hog2.mat
%   : pHOG features we extracted from the CK+ dataset, courtesy of
%    P. Lucey, J. F. Cohn, T. Kanade, J. Saragih, Z. Ambadar, and I. Matthews. 
%    "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for 
%    action unit and emotion-specified expression." In CVPR, 2010.
%
%  (2) http://people.csail.mit.edu/yalesong/fg15/Kassam_AU.mat
%   : AU labels we use to learn AU group structures, courtesy of
%    K. S. Kassam. "Assessment of emotional experience through facial 
%    expression." PhD thesis, Harvard, 2010.
%
% After running the script, you should get an output similar to following:
%   > Experiment with [bcs]
%   > Experiment with [bgcs]
%   > Performance metric: f1
%   > [BCS] 0.636313 (0.040117)
%   > [BGCS] 0.664994 (0.034730)
%   > Performance metric: acc
%   > [BCS] 0.901496 (0.002784)
%   > [BGCS] 0.906557 (0.002651)
%
% Note that the results may look slighly different from our FG'15 paper 
% (Table IV) because below we fix four model parameter values throughout 
% the experiment: comp_rate, gthresh, logchi, and logsigma. 
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bayesian Group-Sparse Compressed Sensing Toolbox  V1.0.0
%
%   Permissions granted under the MIT License (MIT)
%   Copyright (c) 2015 Yale Song (yalesong@csail.mit.edu) 
%
% Please cite the following paper if you end up using the code:
%
%   Yale Song, Daniel McDuff, Deepak Vasisht, and Ashish Kapoor.
%   "Exploiting Sparsity and Co-occurrence Structure for Action Unit
%   Recognition," IEEE FG 2015.
%   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change log
%
% Apr 30, 2015: Initial release
%


%% Initialize parameters for experiments:
%
clear all; clc;

addpath('./lib');
addpath('./helper'); 

mySeed=02319;
rng(mySeed,'twister');

AUs=[1,2,4,5,6,7,9,10,11,12,14,15,16,17,18,20,22,23,24,25,26,27,43,45];

comp_rate = 2.0; % compression rate. <1: compress, >1: expand.
gthresh   = 0.8; % group threshold
logchi    = -1;
logsigma  = -2;  



%% Load CK+ dataset
D = load('./data/CK+_hog2.mat');

y=[D.data.labels]';
y=y(:,AUs); 
data(find(sum(y')==0))=[];

nfolds = numel(unique({D.data.subject})); % leave-one-subject-out
[trn,val,tst] = split_data(D.data,nfolds);
for i=1:nfolds,
    trnFtMat{i} = [trn{i}.features]';
    valFtMat{i} = [val{i}.features]';
    tstFtMat{i} = [tst{i}.features]';
    trnLblMat{i} = [trn{i}.labels]';
    valLblMat{i} = [val{i}.labels]';
    tstLblMat{i} = [tst{i}.labels]';
    trnLblMat{i} = trnLblMat{i}(:,AUs);
    valLblMat{i} = valLblMat{i}(:,AUs);
    tstLblMat{i} = tstLblMat{i}(:,AUs); 
end

% Load Kassam data for learning AU group structure
AU_group_initialization_data = load('./data/Kassam_AU.mat');
group_data = AU_group_initialization_data.data(AUs,:)';



%% Run experiment 
models = {'bcs','bgcs'};
results = cell(numel(models),nfolds);
for i=1:length(models),
    fprintf('Experiment with [%s]\n', models{i});
    for j=1:nfolds,
        switch models{i}
            case 'bcs'
                R = run_bcs(trnFtMat{j},trnLblMat{j},valFtMat{j},tstFtMat{j},...
                    comp_rate,logchi,logsigma);
            case 'bgcs'
                R = run_bgcs(trnFtMat{j},trnLblMat{j},valFtMat{j},tstFtMat{j},...
                    group_data,gthresh,comp_rate,logchi,logsigma);
        end        
        results{i,j} = R;
    end
end


%% get best results
eval_thresh = 0:.05:2;
best_results_f1 = get_best_results( results,'f1',valLblMat,tstLblMat);
best_results_acc = get_best_results( results,'acc',valLblMat,tstLblMat);

