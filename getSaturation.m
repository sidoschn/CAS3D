defaultBasePath = 'D:\';
%defaultBasePath = 'D:\ImageDump01\180208 IO PreDigestion\180208_IO1Stack2_14-40-34';

defaultFileExt = 'C00_xyz Stage Z0000.ome.tif';
%defaultFileExt = 'C00_z0000.ome.tif';
%defaultFileExt = '.tif';

basePath = uigetdir(defaultBasePath, 'Select base path for image detection');
fileExt = inputdlg('Please specify file ending to identify first slice of stack','File extension specification',1,{defaultFileExt});
disp('Searching for images...');
imageFileNameList = getfn(basePath, fileExt);       %Dateien im Ordner und unterordner erfassen

maxChannels = 0;

resultsList = [{'Filename'},{'CAS3d'},{'weightedCAS3d'}];

%checkChannelCount
disp('Checking channels...');
%for i=1:numel(imageFileNameList)
    imageReader = bfGetReader(imageFileNameList{1});
    imageMetaData = imageReader.getMetadataStore();
    %stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    maxChannels = max([maxChannels double(imageMetaData.getPixelsSizeC(0).getValue())]);
%end

channelList = num2str((1:maxChannels)');

[selectedChannel, tf] = listdlg('ListString',channelList,'SelectionMode','single','PromptString','Select the channel to evaluate');

waitBar1 = waitbar(0,'Analyzing Images:')

resultsList = [{'Filename'},{'MaxIntensity'},{'Pixels at max Intensity'},{'Saturation (%)'}];

for i=1:numel(imageFileNameList)
    waitbar((i/numel(imageFileNameList)),waitBar1,['Analyzing Images:' num2str(i) '/' num2str(numel(imageFileNameList))])
    [folderPath, ImageFileName, ImageFileExtension] = fileparts(imageFileNameList{i});
    disp(['Processing: ' ImageFileName])
    imageData = bfopen(imageFileNameList{i});
    imageMetaData = imageData{1,4};
    stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    stackSliceCount = imageMetaData.getPixelsSizeZ(0).getValue();
    stackWidth = imageMetaData.getPixelsSizeX(0).getValue();
    stackHeight = imageMetaData.getPixelsSizeY(0).getValue();
    imagePixelMatrixRaw = zeros(stackHeight,stackWidth,stackSliceCount);
    
    if (selectedChannel > stackChannelCount)
        warning('warning! selected channel exceeds existing channels! Stack is skipped!')
    continue;
    end
    %fill matrix with image values
    for j=1:stackSliceCount
        imagePixelMatrixRaw(:,:,j)=imageData{1,1}{(j*selectedChannel),1};
    end
    [nn, edgesn] = histcounts(imagePixelMatrixRaw);
    
    nTot = stackWidth*stackHeight*stackSliceCount;
    maxInt = edgesn(end);
    nMaxInt = nn(end);
    saturationPercent = 100*nMaxInt/nTot;
    disp(['MaxInt: ' num2str(maxInt) '  nMax: ' num2str(nMaxInt) '   sat: ' num2str(saturationPercent) '%']);
    resultsList = vertcat(resultsList, [{imageFileNameList{i}}, {maxInt},{nMaxInt},{saturationPercent}]);
end
    

cell2csv([basePath '\SaturationResultsSummary.xls'],resultsList);
close(waitBar1);
disp('Analysis complete!');