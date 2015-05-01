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

function R = run_bgcs( trnFtMat, trnLblMat, valFtMat, tstFtMat, group_data, ...
    gthresh, comp, logchi, logsigma, maxiter, verbose, singleton_group )
%RUN_BGCS Summary of this function goes here
%   Detailed explanation goes here

    if ~exist('gthresh','var'), gthresh = .6; end
    if ~exist('comp','var'), comp = 1.0; end
    if ~exist('logchi','var'), logchi = -1; end
    if ~exist('logsigma','var'), logsigma = -2; end
    if ~exist('maxiter','var'), maxiter = 500; end
    if ~exist('singleton_group','var'), singleton_group=true; end
    if ~exist('verbose','var'), verbose = 0; end

    num_labels = size(trnLblMat,2);
    
    % Obtain groups
    groups={};
    for i=1:numel(gthresh),
        G = get_group(group_data, gthresh(i));
        groups{i} = zeros(num_labels,num_labels);
        for m=1:numel(G),
            groups{i}(m,G{m})=1;
        end
        % Optional: add singleton groups
        if singleton_group,
            groups{i} = vertcat(groups{i},eye(num_labels));
        end
        % Check duplicate rows
        dups = [];
        for j=1:size(groups{i},1),
            for k=j+1:size(groups{i},1),
                if isequal(groups{i}(j,:),groups{i}(k,:)),
                    dups(end+1) = k;
                end
            end
        end
        groups{i}(dups,:) = [];        
    end
    
    % Remove duplicate group structures 
    dups = [];
    for i=1:numel(groups),        
        for j=i+1:numel(groups),
            if isequal(groups{i},groups{j}),
                fprintf('%d %d same\n', i,j);
                dups(end+1) = j;
            end
        end        
    end
    gthresh(dups)=[];
    groups(dups)=[];  
    
    R = struct(); cnt=1;
    for i=1:numel(comp), 
        dim_z = ceil(num_labels*comp(i));
        phi = 2*rand(dim_z, num_labels)-1;
        phi = normc(phi); % col-wise normalization
                
        for j=1:numel(logchi),
            for k=1:numel(logsigma),
                chi   = 10^logchi(j);
                sigma = 10^logsigma(k);
                
                if verbose>0,
                    fprintf('BGCS comp_level=%.2f, chi=%.2f, sigma=%.2f',comp(i),chi,sigma);
                end
                tic;

                model = train_bcs(trnFtMat,trnLblMat,phi,sigma,chi,maxiter);
                
                for l=1:numel(groups),  
                    [my.val,mz.val] = test_bgcs(valFtMat,model,maxiter,groups{l});
                    [my.tst,mz.tst] = test_bgcs(tstFtMat,model,maxiter,groups{l});

                    R(cnt).name = 'BGCS';
                    if l==1,
                        R(cnt).model = model; % save model only once
                    else
                        R(cnt).model = [];
                    end
                    R(cnt).params.gthresh = gthresh(l);
                    R(cnt).params.groups = groups{l};
                    R(cnt).params.singleton_groups = singleton_group;
                    R(cnt).params.chi = chi;
                    R(cnt).params.sigma = sigma;
                    R(cnt).params.comp_level = comp(i);                 
                    R(cnt).pred.my.val = my.val; 
                    R(cnt).pred.my.tst = my.tst; 

                    cnt = cnt + 1;
                end
                
                time = toc;
                if verbose>0,
                    fprintf(', time=%.2f\n', time);
                end
            end
        end
    end         
end


function group = get_group( co_occur, thresh )
    group = cell(1,size(co_occur,1));
    for i=1:size(co_occur,1),
        co_occur_i = co_occur(i,:);
        p_co_occur_i = co_occur_i / co_occur(i,i);
        au_above_thresh = find(p_co_occur_i>=thresh);
        group{i} = au_above_thresh;
    end
    
    %{
    AU_cnt = sum(group_data);
    
    group = {};
    for i=1:size(group_data,2),        
        co_occur = sum(group_data(group_data(:,i)>0,:)); % count of co-occurring AUs given au
        p_co_occur = co_occur / AU_cnt(i);
        au_above_thresh = find(p_co_occur>=thresh);
        group{i} = au_above_thresh;
    end 
    %}
end


