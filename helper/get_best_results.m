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

function [ best_results ] = get_best_results( results,perf,valLblMat,tstLblMat )
%GET_BEST_RESULTS Summary of this function goes here
%   Detailed explanation goes here

    eval_thresh=0:.05:2;
    
    best_results=cell(size(results));
    
    % Evaluate. 
    for i=1:size(results,1),
        for j=1:size(results,2),
            R = results{i,j};   
            for k=1:numel(R),
                % Choose eval_thresh on valid split            
                r = cell(1,numel(eval_thresh));            
                for l=1:numel(eval_thresh),
                    ypred = double(R(k).pred.my.val>eval_thresh(l));
                    r{l} = eval_bc(valLblMat{j}(:),ypred(:));
                end
                [~,l_star] = max(cellfun(@(x) getfield(x,perf), r));     
                R(k).params.thresh = eval_thresh(l_star);
                R(k).perf.val = r{l_star};
            end
            results{i,j} = R;
        end
    end

    
    % Find the best result on valid split
    for i=1:size(results,1),
        for j=1:size(results,2),
            if isempty(results{i,j}), 
                continue;
            end
            best_val=-1; best_idx=1;
            for k=1:numel(results{i,j}),
                if getfield(results{i,j}(k).perf.val,perf)>best_val,
                    best_val = getfield(results{i,j}(k).perf.val,perf);
                    best_idx = k;
                end
            end
            R = results{i,j}(best_idx);
            ypred = double(R.pred.my.tst>R.params.thresh);
            R.perf.tst = eval_bc(tstLblMat{j}(:),ypred(:));
            R.perf.tst.confmat_mc = build_confmat_mc(tstLblMat{j},ypred);            
            best_results{i,j} = R;
        end
    end
    
    best_results(find(cellfun(@(x) isempty(x), results)))=[];
    
    print_best_result(best_results,perf);
end


function [ r ] = print_best_result( best_results,perf )
    
    if ~exist('perf','var'),
        perf = 'f1';
    end
    
    fprintf('Performance metric: %s\n', perf);
    for i=1:size(best_results,1),
        if isempty(best_results{i,1}), continue; end
        R = best_results(i,:);
        val = cellfun(@(x) getfield(x.perf.tst,perf),R);
        val(isnan(val)) = 0;
        fprintf('[%s] %f (%f)\n', R{1}.name, mean(val),var(val));
        r(i) = mean(val);
    end
end


function [ cm ] = build_confmat_mc( ytrue, ypred )
%BUILD_CONFMAT_MC Build multi-class confusion matrix

    nclass = size(ytrue,2);   
    cm = zeros(nclass+1,nclass+1);
    
    % Decode
    for i=1:size(ytrue,1),
        ytrue(i,:) = ytrue(i,:).*(1:nclass);
        ypred(i,:) = ypred(i,:).*(1:nclass);
    end
    ytrue(ytrue==0) = nclass+1;
    ypred(ypred==0) = nclass+1;
    
    ytrue = ytrue(:);
    ypred = ypred(:);
    
    for i=1:numel(ytrue),
        cm(ytrue(i),ypred(i)) = cm(ytrue(i),ypred(i))+1;
    end
end


