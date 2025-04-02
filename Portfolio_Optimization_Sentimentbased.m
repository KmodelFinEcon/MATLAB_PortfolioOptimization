%% Portfolio Optimization using Social Performance Measure: Individuals with PhDs
% This example uses the percentage of individuals with PhDs in a company 
% (instead of women on boards) as the social performance measure.
% We build a portfolio that maximizes or minimizes the average PhD percentage 
% while satisfying default and group constraints.

%% Data Setup
% Assume that the percentage of individuals with PhDs per company are as follows.
PhDs = [0.2857; 0.5; 0.3; 0.2857; 0.3077; 0.2727; ...
        0.4167; 0.2143; 0.3; 0.4167; 0.3077];

% Display PhD percentages in a table (assuming Assets is defined elsewhere)
table(PhDs, 'VariableNames', {'PhD_Percentage'}, 'RowNames', Assets(1:11))

%% Create and Configure the Portfolio
% Create portfolio with default constraints.
p = Portfolio('AssetList', Assets(1:11));
p = estimateAssetMoments(p, Data(:,1:11));
p = setDefaultConstraints(p);

% Define group constraints.
% Here, G defines two groups and their lower and upper bounds.
G = [0 1 0 0 1 0 0 0 0 0 0;
     1 0 0 1 0 0 1 0 0 0 0];
LowG = [0.15; 0.25];
UpG = [Inf; 0.5];
p = setGroups(p, G, LowG, UpG);

%% Define Objective Function
% Use the average percentage of individuals with PhDs as the objective.
objectiveFunc = @(x) PhDs' * x;

% Find the portfolio that minimizes the average PhD percentage.
wgt_minPhD = estimateCustomObjectivePortfolio(p, objectiveFunc);
minPhD = objectiveFunc(wgt_minPhD);

% Find the portfolio that maximizes the average PhD percentage.
wgt_maxPhD = estimateCustomObjectivePortfolio(p, objectiveFunc, ...
    "ObjectiveSense", "maximize");
maxPhD = objectiveFunc(wgt_maxPhD);

%% Create Grid for Efficient Frontier
N = 20; % Number of grid points
targetPhD = linspace(minPhD, maxPhD, N);

% Add an inequality constraint to enforce a minimum PhD percentage.
Ain = -PhDs';
bin = -minPhD;
p = setInequality(p, Ain, bin);

% Preallocate cells to store risk, return, and PhD percentages along the frontier.
prsk = cell(N, 1);
pret = cell(N, 1);
pPhD = cell(N, 1);

for i = 1:N
    % Update the PhD constraint for the current target level.
    p.bInequality = -targetPhD(i);
    % Estimate the frontier weights.
    pwgt = estimateFrontier(p, N);
    % Compute portfolio moments.
    [prsk{i}, pret{i}] = estimatePortMoments(p, pwgt);
    % Compute the average PhD percentage.
    pPhD{i} = pwgt' * PhDs;
end

%% Plot the Efficient Frontier
scatter3(cell2mat(prsk), cell2mat(pret), cell2mat(pPhD))
title('Efficient Portfolios')
xlabel('Risk Level')
ylabel('Expected Return')
zlabel('Percentage of PhDs')

%% Plot Contours
nC = 5; % Number of contour plots
minContour = max(pPhD{1}); % Lower bound for PhD percentage on overlapping contours

plotContours(p, minContour, maxPhD, nC, N)

%% Asset Exclusion Example based on PhD Threshold
% Remove the average PhD constraint.
p.AInequality = [];
p.bInequality = [];

% Define a set of thresholds for excluding assets based on PhD percentage.
thresholdPhD = 0.25:0.05:0.40;

% Plot the efficient frontier for different exclusion thresholds.
plotExclusionExample(p, PhDs, thresholdPhD, N)

%% Asset Exclusion and Impact on Return
% Exclude assets with a PhD percentage below 33%.
ub = zeros(p.NumAssets, 1);
ub(PhDs >= 0.33) = 1;
p.UpperBound = ub;

% Estimate the portfolio for a given risk level (e.g., 0.012) after exclusion.
pwgt_exclude = estimateFrontierByRisk(p, 0.012);
ret_exclude = estimatePortReturn(p, pwgt_exclude);

% Return constraints to the original portfolio.
p.UpperBound = [];

%% Incorporate Minimum PhD Constraint into Portfolio
p = addInequality(p, -PhDs', -0.33);

% Estimate the portfolio with the added minimum PhD constraint.
pwgt_avgPhD = estimateFrontierByRisk(p, 0.012);
ret_avgPhD = estimatePortReturn(p, pwgt_avgPhD);

% Remove the PhD inequality constraint.
p.AInequality = [];
p.bInequality = [];

% Calculate return increase due to the average PhD constraint.
ret_increase = (ret_avgPhD - ret_exclude) / ret_exclude;

%% Function Definitions

function [] = plotContours(p, minPhD, maxPhD, nContour, nPort)
    % Plot contour lines for different minimum PhD percentage levels.
    
    % Define a set of PhD percentage levels for contour plotting.
    contourPhD = linspace(minPhD, maxPhD, nContour+1);
    
    % Open a new figure and hold on to plot multiple lines.
    figure;
    hold on
    labels = strings(nContour+1, 1);
    
    % Plot the efficient frontier for each PhD level.
    for i = 1:nContour
        p.bInequality = -contourPhD(i);
        plotFrontier(p, nPort);
        labels(i) = sprintf("%6.2f%% PhDs", contourPhD(i)*100);
    end
    
    % Plot the original mean-variance frontier (no PhD restriction).
    p.AInequality = [];
    p.bInequality = [];
    plotFrontier(p, nPort);
    labels(nContour+1) = "No PhD restriction";
    
    legend(labels, 'Location', 'northwest')
    hold off
end

function [] = plotExclusionExample(p, PhDs, thresholdPhD, nPort)
    % Plot the efficient frontier when excluding assets below varying PhD thresholds.
    
    nT = length(thresholdPhD);
    figure;
    hold on
    labels = strings(nT+1, 1);
    
    for i = 1:nT
        ub = zeros(p.NumAssets, 1);
        % Only include assets with a PhD percentage above the threshold.
        ub(PhDs >= thresholdPhD(i)) = 1;
        p.UpperBound = ub;
        plotFrontier(p, nPort);
        labels(i) = sprintf("%6.2f%% PhDs", thresholdPhD(i)*100);
    end
    
    % Plot the frontier with no exclusion.
    p.UpperBound = [];
    plotFrontier(p, nPort);
    labels(nT+1) = "No PhD restriction";
    
    legend(labels, 'Location', 'northwest')
    hold off
end