%HRP De Prado portfolio Optimization using cluster analysis model implementation by K.Tomov

function [wHRP, wMV, info] = computeHRP(assetRetn, varargin)

%  [wHRP, wMV, info] = computeHRP(assetRetn, Name,Value,...)
%
%  Inputs:
%    assetRetn    T×N matrix of asset returns
%
%  Name‑Value Parameters:
%    'MaxClusters'   maximum number of clusters in HRP (default: 4)
%    'CovShrinkage'  true/false to apply Ledoit-Wolf shrinkage (default: true)
%    'ClusterMethod' linkage method, e.g. 'single','ward' (default: 'single')
%
%  Outputs:
%    wHRP   N×1 HRP weights (sum to 1)
%    wMV    N×1 Global minimum‑variance weights
%    info   struct containing:
%    .Sigma       covariance matrix
%    .C           correlation matrix
%    .link        linkage result
%    .order       quasi‑diag order indices
%    .clusters    cluster membership vector
%    .RC_HRP      risk contributions under HRP
%    .RC_MV       risk contributions under MV

  %inputs
  p = inputParser;
  addParameter(p,'MaxClusters',4,@(x)isscalar(x)&&x>=1);
  addParameter(p,'CovShrinkage',true,@islogical);
  addParameter(p,'ClusterMethod','single',@ischar);
  parse(p,varargin{:});
  K   = p.Results.MaxClusters;
  doShrink = p.Results.CovShrinkage;
  method  = p.Results.ClusterMethod;

  % stat computation
  [~,N] = size(assetRetn);
  mu     = mean(assetRetn);

  % Covariance construction (with optional shrinkage)
  if doShrink
    Sigma = ledoitWolf(assetRetn);
  else
    Sigma = cov(assetRetn);
  end
  C = corrcov(Sigma);

  % Distance matrix and linkage
  distMat = squareform( pdist(C,'euclidean') );
  link    = linkage(distMat,'method',method);

  % Quasi‑diagonal ordering
  order = getQuasiDiagonal(link,N);

  % Cluster assignment
  clusters = cluster(link,'Maxclust',K);

  % HRP weights & risk contributions
  wHRP = zeros(N,1);
  W     = zeros(N,K);
  parfor i=1:K
    idx  = (clusters==i);
    subS = Sigma(idx,idx);
    W(idx,i) = riskBudgetingPortfolio(subS);
  end
  covClust = W' * Sigma * W;
  wB       = riskBudgetingPortfolio(covClust);
  wHRP     = W * wB;
  RC_HRP   = marginalRiskContrib(wHRP,Sigma);

  % Minimum‑variance (minimization function)
  pObj = Portfolio('AssetMean',mu,'AssetCovar',Sigma);
  pObj = setDefaultConstraints(pObj);
  wMV  = estimateFrontierLimits(pObj,'min');
  RC_MV = marginalRiskContrib(wMV,Sigma);

  % Package info
  info.Sigma    = Sigma;
  info.C        = C;
  info.link     = link;
  info.order    = order;
  info.clusters = clusters;
  info.RC_HRP   = RC_HRP;
  info.RC_MV    = RC_MV;

  % Plot
  plotResults(C,link,order,clusters,wMV,wHRP,RC_MV,RC_HRP);
end

function Sigma = ledoitWolf(X)
  % Ledoit–Wolf shrinkage covariance estimator
  [T,N] = size(X);
  S = cov(X);
  mu = trace(S)/N;
  F = mu * eye(N);
  varS = 0;
  for t=1:T
    dt = (X(t,:)'*X(t,:) - S);
    varS = varS + sum(sum(dt.^2));
  end
  varS = varS/T;
  beta = min(varS / sum(sum((S-F).^2)),1);
  Sigma = beta*F + (1-beta)*S;
end

function order = getQuasiDiagonal(link,N)
  % Iterative ordering of leaf nodes
  % Each merge, take children in order of decreasing linkage distance
  children = num2cell((1:N)');
  for i=1:size(link,1)
    c1 = children{link(i,1)};
    c2 = children{link(i,2)};
    % compare max correlations within clusters to decide order
    m1 = max(corrInterp(c1));
    m2 = max(corrInterp(c2));
    if m1>m2
      children{N+i} = [c1;c2];
    else
      children{N+i} = [c2;c1];
    end
  end
  order = children{end};
  
  function m = corrInterp(idx)
    m = abs(link( max(idx),3 ));
  end
end

function RC = marginalRiskContrib(w,Sigma)
  % Marginal risk contributions
  portVar = w' * Sigma * w;
  mRisk   = Sigma*w;
  RC      = w .* mRisk  / sqrt(portVar);
end

function plotResults(C,link,order,clusters,wMV,wHRP,RC_MV,RC_HRP)
  clf;
  tiledlayout(2,2,'TileSpacing','Compact');

  % 1. Dendrogram + clusters
  nexttile([1 2]);
  dendrogram(link,'Reorder',order,'ColorThreshold','default');
  title('Hierarchical Clustering');
  xlabel('Asset Index'); ylabel('Linkage Distance');

  % 2. Clustered correlation heatmap
  nexttile;
  imagesc(C(order,order));
  axis square; colorbar;
  title('Reordered Correlation Matrix');
  xticks(1:length(order)); yticks(1:length(order));
  xticklabels(order); yticklabels(order);

  % 3. Weights bar
  nexttile;
  bar([wMV(order), wHRP(order)]);
  legend({'MinVar','HRP'},'Location','Best');
  title('Portfolio Weights (Sorted)'); xlabel('Asset (by order)');

  % 4. Risk Contributions
  nexttile;
  bar([RC_MV(order), RC_HRP(order)]);
  legend({'RC\_MV','RC\_HRP'},'Location','Best');
  title('Marginal Risk Contributions'); xlabel('Asset (by order)');
end
