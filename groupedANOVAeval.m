[~,~, rawData] = xlsread('CAS3dResultsSummary4.3Abridged.xlsx');
clear temp1
clear temp2

Filename = string(rawData(2:end,1));
SetnamePattern = '\\[0123456789][0123456789][0123456789]\\';
SetnamePos = cell2mat(regexp(Filename,SetnamePattern));
SetnameLength = 3;
SetNames = extractBetween(Filename,SetnamePos+1,SetnamePos+SetnameLength);

columnHeaders = rawData(1,2:end);


numericDataset = rawData(2:end,2:end);
numericDataset(strcmp(string(numericDataset),"NaN")) = {NaN};
numericDataset = cell2mat(numericDataset);
uSetNames = unique(SetNames);

GroupedData = uSetNames';

maxSetSize = 0;


for i = 1:length(uSetNames)
    setSize = sum(strcmp(SetNames,uSetNames(i)));
    if (setSize>maxSetSize)
        maxSetSize = setSize;
    end
end
% setData = cell(maxSetSize,8,length(uSetNames));
setData = NaN([maxSetSize 9 length(uSetNames)]);

for i = 1:length(uSetNames)
    
    partialSetData = numericDataset(strcmp(SetNames,uSetNames(i)),:);
    setData(1:size(partialSetData,1),:,i) = partialSetData;
end
% setData(isempty(setData)) = {NaN};

% setDataMat = cell2mat(setData);

meansPerSet = nanmean(setData,1);
stdDevsPerSet = nanstd(setData,0,1);

targetParameters = [];

for i=targetParameters
    figure('Name',columnHeaders{i})
    
    boxplot(squeeze(setData(:,i,:)),uSetNames)
    
end

%group groups

groupGroupTitles = [{'control'},{'PN+3HB'},{'PN+gluc'},{'ungrouped'}];
groupGroups = [{'204'},{'217'},{'237'},{'269'},{'286'};{'214'},{'234'},{'245'},{'259'},{'283'};{'193'},{'208'},{'219'},{'240'},{'258'};{'262'},{'290'},{''},{''},{''}];

% [bIsGrouped, posInGroup] = ismember(uSetNames(:),groupGroups(:));
% ungoupedOnes = uSetNames(~bIsGrouped);
setSize = zeros(size(groupGroups,1),1);
for i = 1:size(groupGroups,1)
    for j = 1:size(groupGroups,2)
        setSize(i) = setSize(i)+size(numericDataset(strcmp(SetNames,groupGroups(i,j)),:),1);
        
    end
end

setdGroupedData = NaN([max(setSize) 9 size(groupGroups,1)]);

for i = 1:size(groupGroups,1)
    partialSetData = [];
    for j = 1:size(groupGroups,2)
        partialSetData = vertcat(partialSetData,numericDataset(strcmp(SetNames,groupGroups(i,j)),:));
        
    end
    setdGroupedData(1:size(partialSetData,1),:,i) = partialSetData;
end

groupGroupTitles = [{['control n=' num2str(setSize(1))]},{['PN+3HB n=' num2str(setSize(2))]},{['PN+gluc n=' num2str(setSize(3))]},{['ungrouped n=' num2str(setSize(4))]}];

targetParameters = [1 2 3 9];

cTot = [];

fig = gobjects(numel(targetParameters));

for i=targetParameters
%     figure('Name',columnHeaders{i})
%     boxplot(squeeze(setdGroupedData(:,i,:)),groupGroupTitles)
    [p, t, stats] = anova1(squeeze(setdGroupedData(:,i,:)),groupGroupTitles,'off');
    %title(columnHeaders{i});
    [c,m,h,nms] = multcompare(stats,'Display','off');
    %close(h);
    %title(columnHeaders{i});
    cTot = cat(3,cTot,c);
    
    fig(i) = figure('Name',columnHeaders{i});
    boxplot(squeeze(setdGroupedData(:,i,:)),groupGroupTitles);
    ylabel(columnHeaders{i});
    sigPairs = {};
    sigPs = [];
    for j = 1:size(c,1)
         if c(j,6)<0.05
            sigPairs = [sigPairs,[c(j,1),c(j,2)]];
            sigPs = [sigPs,c(j,6)];
         end
    end
    sigstar(sigPairs,sigPs);
    
end

