
%CAS3d Tool Version 4.43
%Dominik Schneidereit
%Institute of Medical Biotechnology
%2019


%4.43
%- added more intuitive file selection
%- removed channel selection when only one channel is detected
%- tied the CSV cell separator character to system default values.
%- added basepath retention during multiple analysis from the same launcher
%- re-adjusted orientation and elevation outputs to be between
%  0 and 180 or 90 and -90° respectively
%- improved consistency of orientation nomenclature (eg. avoiding 40°
%  angles for -40° elevated patterns
%- changed nomenclature from CAS3d to CAS3D 
%- added sub-folder to output image nomenclature
%- improved image intensity adaption to saturate the lowest and highest 1%
%- changed output graph to show background percent
%- retain filter string from previous eval

%4.42
%added isotropic pattern handling

%set defaults
clearvars -except src basePath filterString
wCount = 0;
defaultBasePath = 'C:\';
%defaultFileExt = 'C00_xyz Stage Z0000.ome.tif';
defaultFileExt = '.tif';
resultsFileName = 'CAS3dResultsSummary';

%query path and file extension
global basePath;

%for base path retention in during multiple runs from the same launcher
%window
if ~isempty(basePath) && not(isscalar(basePath))
    defaultBasePath = basePath;
    basePath = 0;
end

% %initialize process abortion functionality
% global bAbortEval;
% abortMessage = "Evaluation was aborted";

%basePath = 'E:\DATEN\Messdaten\Omero_Dropbox\Schneidereit\SoftwareDevelopment\MATLAB no source control\cas3dTool\TestImagesAndStacks\NoisySineWaveStacks';
basePath = uigetdir(defaultBasePath, 'Select base path for image detection (subfolders will also be searched)');
if basePath == 0
    basePath = defaultBasePath;
    return
end

global filterString;
global fileList;
%fileList = getfn(basePath,'Specimen_RawData');
fileList = getfn(basePath, defaultFileExt);
global filteredFileList;
filteredFileList = fileList(1);
global bottomOffset;
bottomOffset = 50;
global bOK
bOK = 0;
%filterString = "";
global bSaveFigure;
bSaveFigure = 1;
initiallist = fileList;

%make the user filter for and select the data sets for evaluation

f = figure;
f.NumberTitle = 'off';
f.Name = 'Select data set(s) to evaluate';
f.ToolBar = 'none';
f.MenuBar = 'none';
oldPos = f.Position;
f.Position = [oldPos(1),oldPos(2),700,420];
f.SizeChangedFcn = @adaptTableSize_callback;
% maybe better use listdlg?
uil = uicontrol(f, 'Style', 'listbox');
uil.String = initiallist;
uil.Max = length(fileList);
uil.Position = [0 bottomOffset f.Position(3) f.Position(4)-bottomOffset];
uil.Callback = @listSelection_callback;

filtField = uicontrol(f,'Style','edit','Callback', @filtButton_callback, 'String',filterString);
filtField.Position = [filtField.Position(1)+60 filtField.Position(2) filtField.Position(3)+100 filtField.Position(4)];

saveCheckBox = uicontrol(f,'Style','checkbox','Callback',@saveFigureCheckbox_callback,'String', 'Save Figures','Value',1);
saveCheckBox.Position = [filtField.Position(1)+300 filtField.Position(2) filtField.Position(3)+100 filtField.Position(4)];
%filtField = uicontrol(f,'Style','edit');

%find the java object
%jObj = findjobj(filtField,'nomenu');
%set the keytypedcallback
%set(jObj,'KeyTypedCallback',@myKeyTypedCallbackFcn)

%filtIndic = uicontrol(f,'Style','text', 'Position', [filtField.Position(1)+160 filtField.Position(2)-3 200 filtField.Position(4)+10], 'String', 'Current Filter:');
filtTag = uicontrol(f,'Style','text', 'Position', [filtField.Position(1)-80 filtField.Position(2)-3 80 filtField.Position(4)], 'String', 'Filter phrase:');

bfilt = uicontrol(f,'Style','pushbutton','Callback',@filtButton_callback);
bfilt.String = 'Apply';
bfilt.Position = [filtField.Position(1)+160 filtField.Position(2) 60 filtField.Position(4)];



bcfilt = uicontrol(f,'Style','pushbutton','Callback',@clearFiltButton_callback);
bcfilt.String = 'Clear';
bcfilt.Position = [filtField.Position(1)+220 filtField.Position(2) 60 filtField.Position(4)];

bc = uicontrol(f,'Style','pushbutton','Callback',@closebutton_callback);
bc.Position =  [filtField.Position(1)+520 filtField.Position(2) 60 filtField.Position(4)];
bc.String = 'OK';

if not(isempty(filterString))
   filtButton_callback(bfilt,'');
end
drawnow;
waitfor(f);

if bOK == 0
    disp('Process aborted');
    return
end

fileList = filteredFileList;

imageFileNameList = fileList;
%imageFileNameList = {'E:\DATEN\Messdaten\Omero_Dropbox\Schneidereit\SoftwareDevelopment\MATLAB no source control\cas3dTool\TestImagesAndStacks\NoisySineWaveStacks\idealSineWave_00Deg_f0_5Stack.tifNoise1.tif'};

% this part is depreceated and replaced by better GUI
% fileExt = inputdlg('Please specify a file ending to identify the first slice of a stack','File extension specification',1,{defaultFileExt});
% if isempty(fileExt)
%     exit
% end
% disp('Searching for images...');
% imageFileNameList = getfn(basePath, fileExt);

%load 3d sobel kernels
load('sobel3x.mat');
load('sobel3y.mat');
load('sobel3z.mat');

% reform sobel matrix to larger radius
% this feature was disabled as it increases the noise resilience just a bit
% at a huge cost of rotational accuracy!!!
% sobelSize = 5;
%
% sobel3x = imresize3(sobel3x, [sobelSize,sobelSize,sobelSize], "method", "lanczos3");
% sobel3y = imresize3(sobel3y, [sobelSize,sobelSize,sobelSize], "method", "lanczos3");
% sobel3z = imresize3(sobel3z, [sobelSize,sobelSize,sobelSize], "method", "lanczos3");

%check if logfile already exists and if so, add an increasing numeral to
%the file name to keep it unique, then start a log file
if isfile([basePath '\' resultsFileName '.csv'])
    newResultsFileName = resultsFileName;
    i = 0;
    while isfile([basePath '\' newResultsFileName '.csv'])
        newResultsFileName =  [resultsFileName sprintf('%03u' ,i)];
        i=i+1;
    end
    resultsFileName = newResultsFileName;
end

sCs = getSystemCSVseparator();
% '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n'
resultsFileHeaderFormat= ['%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s' sCs '%s\n'];
% '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'
resultsFileDataFormat= ['%s' sCs '%.4f' sCs '%.4f' sCs '%.4f' sCs '%.4f' sCs '%.4f' sCs '%.4f' sCs '%.4E' sCs '%.4E' sCs '%.4f\n'];

resultsFileID = fopen([basePath '\' resultsFileName '.csv'],'a');
resultsHeader = [{'Filename'},{'CAS3D (-)'},{'intWeightedCAS3D (-)'},{'MainOrientation (°)'},{'MainElevation (°)'},{'WeightedAngularDeviation (°)'},{'AngularDeviation (°)'},{'Saturation (%)'},{'Background (%)'}, {'Mean SL (µm)'}];
fprintf(resultsFileID,resultsFileHeaderFormat,resultsHeader{:});

%checkChannelCount in first found image
disp('Checking channels...');

imageReader = bfGetReader(imageFileNameList{1});
imageMetaData = imageReader.getMetadataStore();
maxChannels =double(imageMetaData.getPixelsSizeC(0).getValue());


%query analysis channel
if maxChannels>1
    channelList = num2str((1:maxChannels)');
    [selectedChannel, tf] = listdlg('ListString',channelList,'SelectionMode','single','PromptString','Select the channel to evaluate');
else
    selectedChannel = 1;
end

%run analysis for each detected file in FileList
waitBar1 = waitbar(0,'Analyzing Images:');

%if run from the launcher, enable the "working state" circle gif
try
    src.Parent.Children(end-1).Visible = 'on';
catch
end

for i=1:numel(imageFileNameList)
    waitbar(((i-1)/numel(imageFileNameList)),waitBar1,['Analyzing Images:' num2str(i) '/' num2str(numel(imageFileNameList))]);
    [folderPath, ImageFileName, ImageFileExtension] = fileparts(imageFileNameList{i});
    disp(['Processing: ' folderPath(numel(basePath)+2:end) '_' ImageFileName])
    
    imageData = bfopen(imageFileNameList{i});
    imageMetaData = imageData{1,4};
    stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    stackSliceCount = imageMetaData.getPixelsSizeZ(0).getValue();
    stackWidth = imageMetaData.getPixelsSizeX(0).getValue();
    stackHeight = imageMetaData.getPixelsSizeY(0).getValue();
    physPixSizeX = imageMetaData.getPixelsPhysicalSizeX(0).value().doubleValue();
    physPixSizeY = imageMetaData.getPixelsPhysicalSizeY(0).value().doubleValue();
    physPixSizeZ = imageMetaData.getPixelsPhysicalSizeZ(0).value().doubleValue();
    %     physPixUnitsX = char(imageMetaData.getPixelsPhysicalSizeX(0).unit().getSymbol());
    %    physPixSizeX = imageMetaData.getPixelsPhysicalSizeX(0).getValue();
    %    physPixSizeY = imageMetaData.getPixelsPhysicalSizeY(0).getValue();
    %    physPixSizeZ = imageMetaData.getPixelsPhysicalSizeZ(0).getValue();
    physPixUnitsX = "micron";
    
%     %checkpoint to abort
% if bAbortEval ==1
%     disp(abortMessage)
%     return;
% end

    
    voxelHeightToWidthFactor = physPixSizeZ/physPixSizeX;
    
    %create empty matrix with dimensions of image stack
    imagePixelMatrixRaw = zeros(stackHeight,stackWidth,stackSliceCount);
    
    if (selectedChannel > stackChannelCount)
        warning('warning! selected channel exceeds existing channels! Stack is skipped!')
        wCount = wCount+1;
        continue;
    end
    
    %fill matrix with image values
    for j=1:stackSliceCount
        imagePixelMatrixRaw(:,:,j)=imageData{1,1}{(j*selectedChannel),1};
    end
    
    %free up RAM
    clear imageData;
    
    % denoise with 3d gaussian filter with a sigma of 1/1000th of
    % image pixel number
    % in 4.43 this is changed to remove the brightest pixelsfirst, then
    % normalize again
    
    upperThresh = 0.99999;
    
    imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw,round(size(imagePixelMatrixRaw,1))/1000);
    %imagePixelMatrix = imagePixelMatrixRaw;
    imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
    
    imagePixelMatrix = imadjustn(imagePixelMatrixNorm);
    
   
    
    
%     median(imagePixelMatrixNorm(:))
%     return
%     
    %new function integrated in 4.43 to replace thresholding, instead using
    %the pixel count
%     imagePixelMatrix= removeBrightestVoxels(imagePixelMatrixNorm, upperThresh);
%     
%     imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
%     
%     imagePixelMatrix = imagePixelMatrixNorm;
     
     lowerThresh = (min(imagePixelMatrix,[],'all')+0.1);
    
    saturation = 100*sum((imagePixelMatrix>upperThresh), 'all')/numel(imagePixelMatrix);
    background = 100*sum((imagePixelMatrix<lowerThresh), 'all')/numel(imagePixelMatrix);
    
    if (saturation == 0 && background == 0)
        warning('warning! Homogeneous intensity across whole stack detected, the Stack is skipped!')
        continue;
    end
    
    disp(['Saturation: ' num2str(saturation) '%  Background: ' num2str(background) '%'])
    
%     %checkpoint to abort
% if bAbortEval ==1
%     disp(abortMessage)
%     return;
% end

    %as the brightest pixels were already removed, this was changed to only
    % remove the background
    %imagePixelMatrixRaw=imagePixelMatrix .*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imagePixelMatrixRaw=imagePixelMatrix .*(((lowerThresh<imagePixelMatrix)));
    imagePixelMatrixRaw=imagePixelMatrix;
    
    %geting main direction and mean Sarcomere Length by 3d fft analysis
    fftw('dwisdom',[]);
    fftw('planner','estimate');
    imagefft = fftn(imagePixelMatrixRaw);
    
    imageffts = fftshift(imagefft,2);
    imageffts = fftshift(imageffts,1);
    imageffts = fftshift(imageffts,3);
    imagerffts = abs(real(imageffts));
    filtImagerffts = imgaussfilt3(imagerffts,round(size(imagerffts,1)/100));%%%%% ## this is probably tha bad boy
    bwThresh = max(filtImagerffts(:));
    %bwThresh = max(filtImagerffts,[],'all');
    iterations = 0;
    BWfft = imbinarize(filtImagerffts,bwThresh);
    stats = regionprops3(BWfft);
    
    %this is a debugging feature
    %saveFFTasTiff(filtImagerffts);
    
    bwThreshMod = 1.1;
    bwThreshModMod = 0.1;
    %iteration limit reduced from 1000 to speed up the detection
    maxIterationLimit = 200;
    
    bIsIsotropic = 0;
    disp('Finding dominant pattern and its orientation through iterative Fourier transform analysis');
    threshsList = [];
    nPeaksList = [];
    while(~(size(stats,1)==3))
        fprintf('.');
        bwThresh = bwThresh / bwThreshMod;
        BWfft = imbinarize(filtImagerffts,bwThresh);
        stats = regionprops3(BWfft);
        iterations = iterations+1;
        threshsList = [threshsList bwThresh];
        nPeaksList = [nPeaksList size(stats,1)];
        %disp([bwThresh size(stats,1)]);
        if (size(stats,1)>3)
            bwThresh = bwThresh*bwThreshMod;
            bwThreshMod = 1+((bwThreshMod-1)*bwThreshModMod);
        else
        end
        
        if 0==mod(iterations,40)
            disp(' ');
        end
        
        %in case you pass the max iteration limit, discard the image stack
        if iterations > maxIterationLimit
            break;
        end
        
    end
    
    if (iterations > maxIterationLimit)
        warning('warning! Unable to detect secondary peaks in fourier Transform! Assuming isotropic data!')
        bIsIsotropic = 1;
        wCount = wCount+1;
    end
    
    if bIsIsotropic == 0
        %getting centroid vectors of secondary peaks by picking one of the
        %longer vectors (major peak centroid vector length should be zero)
        
        xVectors = stats.Centroid(:,1)-((size(imagefft,1)/2)+1);
        yVectors = stats.Centroid(:,2)-((size(imagefft,2)/2)+1);
        %zVectors = (stats.Centroid(:,3)-((size(imagefft,3)/2))-1)*voxelHeightToWidthFactor;
        %ratio is unneccessary due to how FFT works and is removed
        zVectors = stats.Centroid(:,3)-((size(imagefft,3)/2))-1;
        
        magnitudefftVectors = sqrt((xVectors.^2)+(yVectors.^2)+(zVectors.^2));
        
        % identifies the primary peak by its (usually 0) short distance
        % from the center
        [~,centerPeakIndex] = min(magnitudefftVectors);
        
        xVector = 0;
        yVector = 0;
        zVector = 0;
        
        %uses the righthand secondary peak for further eval, keeping
        %orientation and elevation nomenclature consistent
        for j=1:numel(magnitudefftVectors)
           if not(j==centerPeakIndex)
               if xVectors(j)>=xVector
                    xVector = xVectors(j);
                    yVector = yVectors(j);
                    zVector = zVectors(j);
               end
           end
        end
        
%                 % picks the vector with higher magnitude (length) to
%                 % differentiate between secondary and primary peaks, this
%                 % was replaced in 4.43 for a more refined method.
%                 if magnitudefftVectors(1)>magnitudefftVectors(2)
%                     xVector = xVectors(1);
%                     yVector = yVectors(1);
%                     zVector = zVectors(1);
%         
%                 else
%                     xVector = xVectors(2);
%                     yVector = yVectors(2);
%                     zVector = zVectors(2);
%         
%                 end
               
    else
        intensityAtRadius = filtImagerffts(round(size(filtImagerffts,1)/2):end,round(size(filtImagerffts,2)/2),round(size(filtImagerffts,3)/2));
        figure
        plot(intensityAtRadius);
        [peaks,locs] = findpeaks(intensityAtRadius);
        [value,index] =max(peaks);
        xVector = locs(index);
        yVector = 0;
        zVector = 0;
        %[xVector,yVector,zVector] = processIsotropicData(filtImagerffts);
        
    end
    
    mainDirection = atan2(yVector,xVector);
    mainElevation = atan2(sqrt((xVector.^2)+(yVector.^2)),zVector);
    mainDirectionDeg = rad2deg(atan2(yVector,xVector));
    mainElevationDeg = rad2deg(atan2(sqrt((xVector.^2)+(yVector.^2)),zVector));
    magnitudefftVector = sqrt((xVector.^2)+(yVector.^2)+(zVector.^2));
    meanSL = physPixSizeX*size(imagefft,1)/magnitudefftVector;
    
    %re-ranging orientatnion to in between 0 and 180°
    %disp(mainDirectionDeg)
    mainDirectionDeg = reRangeDir(mainDirectionDeg);
    %disp(mainDirectionDeg)
    %re-ranging elevation to in between 0 and 90°
    %disp(mainElevationDeg)
    mainElevationDeg = reRangeElev(mainElevationDeg);
    %disp(mainElevationDeg)
    
    disp(' ');
    disp(['Iterations neccessary to find secondary peaks: ' num2str(iterations)]);
    disp(strjoin(['Detected main orientation ' num2str(mainDirectionDeg) '° and elevation ' num2str(mainElevationDeg) '°' ' with mean SL: ' num2str(meanSL) ' ' physPixUnitsX]));
    
%     %checkpoint to abort
% if bAbortEval ==1
%     disp(abortMessage)
%     return;
% end

    
    % create norm vector with main direction and elevation heading
    xyzMainVector = zeros(1,3);
    xyzMainVector(1) = sin(mainElevation)*cos(mainDirection); %vector x
    xyzMainVector(2) = sin(mainElevation)*sin(mainDirection); %vector y
    xyzMainVector(3) = cos(mainElevation); %vector z
    
    %     % denoise with 3d gaussian filter with a sigma of 1/1000th of
    %     % image pixel number
    %
    %     imagePixelMatrix = imgaussfilt3(imagePixelMatrixRaw,round(size(imagePixelMatrixRaw,1))/1000);
    %     %     imagePixelMatrix = imagePixelMatrixRaw;
    %     imagePixelMatrixNorm = imagePixelMatrix./max(imagePixelMatrix,[],'all');
    %     imagePixelMatrix = imagePixelMatrixNorm;
    %     lowerThresh = (min(imagePixelMatrix,[],'all')+0.1);
    %     upperThresh = 0.999;
    %     saturation = 100*sum((imagePixelMatrix>upperThresh), 'all')/numel(imagePixelMatrix);
    %     background = 100*sum((imagePixelMatrix<lowerThresh), 'all')/numel(imagePixelMatrix);
    %
    %     disp(['Saturation: ' num2str(saturation) '%  Background: ' num2str(background) '%'])
    %
    %create sobel vectors for pixels within the threshold limits
    imageSobelX = imfilter(imagePixelMatrix,sobel3x,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelY = imfilter(imagePixelMatrix,sobel3z,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)));
    imageSobelZ = imfilter(imagePixelMatrix,sobel3y,'conv','replicate').*(((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)))./voxelHeightToWidthFactor;
    
    %get Magnitudes of sobel vectors and set zero length vectors to NaN
    %in order to exclude them from evaluation
    sobelMagnitudes = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    sobelMagnitudes(sobelMagnitudes==0) = NaN;
    
    %normalize sobel vectors to a length of 1 (to enable a unweighted
    %determination of directionalities)
    
    imageSobelX = imageSobelX./sobelMagnitudes;
    imageSobelY = imageSobelY./sobelMagnitudes;
    imageSobelZ = imageSobelZ./sobelMagnitudes;
    
    %-----
    %here directionality and magnitude of each pixel is determined,
    %-----
    
    %create empty matrix to hold normal vectors
    imageVectorMatrix = zeros(stackHeight,stackWidth,stackSliceCount,4);
    
    %calculate direction angle
    imageVectorMatrix(:,:,:,1) = rad2deg(atan2(imageSobelY,imageSobelX));
    
    %calculate magnitude
    imageVectorMatrix(:,:,:,3) = sqrt((imageSobelX.^2)+(imageSobelY.^2)+(imageSobelZ.^2));
    
    %calculate elevation angle
    imageVectorMatrix(:,:,:,2) = rad2deg(atan2(sqrt((imageSobelX.^2)+(imageSobelY.^2)),imageSobelZ));
    
    %calculate cosines of divergence angle from main direction
    imageVectorMatrix(:,:,:,4) = abs(imageSobelX.*xyzMainVector(1)+imageSobelY.*xyzMainVector(2)+imageSobelZ.*xyzMainVector(3));
    
    %calculate the cosine angle sum of the image stack
    CAS3d = nanmean(imageVectorMatrix(:,:,:,4),'all');
    
    %calculate the intensity weighted cosine angle sum of the image stack
    CAS3dIntWeight = nansum((imageVectorMatrix(:,:,:,4).*imagePixelMatrix),'all')/nansum((((lowerThresh<imagePixelMatrix)&(imagePixelMatrix<upperThresh)).*imagePixelMatrix),'all');
    
    disp(['CAS3D = ' num2str(CAS3d) ' wCAS3D = ' num2str(CAS3dIntWeight)]);
    disp('');
    
%     %checkpoint to abort
% if bAbortEval ==1
%     disp(abortMessage)
%     return;
% end

    
    %----
    %here the output plot is generated and saved
    %----
    if exist('resFig')
        delete(resFig);
    end
    resFig = figure('Name', ImageFileName, 'MenuBar', 'none', 'Toolbar', 'none', 'NumberTitle', 'off');
    %sub plot 1 shows the intensity weighted histogram of divergence from
    %the main direction in °
    sp0 = subplot(2,2,1);
    [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))).*imagePixelMatrix);
    %if the sample is too perfect, there will be only one histogram bin, dont to a peak analysis in that case
    if numel(divY)>2
        findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
        [peakInts, weightedPeakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
    else
        weightedPeakLocs = [];
    end
    
    if (numel(weightedPeakLocs)==0)
        weightedPeakLocs = [NaN NaN NaN];
        peakInts = [NaN NaN NaN];
    end
    displayText0 = {[' Major Divergence: ' num2str(weightedPeakLocs(1)) '°']};
    if exist('displayText0item')
        delete(displayText0item);
    end
    
    
    displayText0item = text(weightedPeakLocs(1),peakInts(1),displayText0);
    xlabel('divergence angle (°)')
    ylabel('binned signal intensity (-)')
    
    %subplot 2 shows the histogram of divergence from main direction in °
    sp1 = subplot(2,2,2);
    [divY, divX] = histcounts(rad2deg(acos(imageVectorMatrix(:,:,:,4))));
    if numel(divY)>2
        findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4)
        [peakInts, peakLocs] = findpeaks(divY,divX(1:end-1),'MinPeakProminence',mean(divY)/4);
    else
        peakLocs = [];
    end
    
    if (numel(peakLocs)==0)
        peakLocs = [NaN NaN NaN];
        peakInts = [NaN NaN NaN];
    end
    displayText1 = {[' Major Divergence: ' num2str(peakLocs(1)) '°']};
    if exist('displayText1item')
        delete(displayText1item);
    end
    displayText1item = text(peakLocs(1),peakInts(1),displayText1);
    xlabel('divergence angle (°)')
    ylabel('binned pixel counts (-)')
    
    %subplot 3 shows the center slice of the image stack with the cosine
    %angle divergence from the main direciton of each pixel color coded
    %from 0-1. Pixels that are not used in the evaluation are set to 0
    sp2 = subplot(2,2,3);
    imagesc(imageVectorMatrix(:,:,round(stackSliceCount/2),4));
    axis off;
    title('cosines of center slice');
    colorbar('eastoutside');
    %save the center slice as a tiff file
    %imwrite(imagePixelMatrix(:,:,round(stackSliceCount/2)), [basePath '\' ImageFileName '.centerSlice.png'],'png');
    
    %subplot 4 displays a summary of the detected parameters
    sp3 = subplot(2,2,4);
    cla(sp3)
    axis off;
    displayText = {['Main orientation: ' num2str(mainDirectionDeg) '°'],['Main elevation: ' num2str(mainElevationDeg) '°'],['CAS3D: ' num2str(CAS3d)],['Weighted CAS3D: ' num2str(CAS3dIntWeight)],['Background: ' num2str(background) '%'], strjoin(['mean SL: ' num2str(meanSL) ' ' physPixUnitsX])};
    if exist('textItem')
        delete(textItem);
    end
    textItem = text(0,0.5,displayText);
    
    subFolderUID = [replace(folderPath(numel(basePath)+2:end),'\','_') '_'];
    
    %the plot is saved in the base folder with the name of the image stack
    if bSaveFigure == 1
        print(resFig,[basePath '\' subFolderUID ImageFileName '.dirDist.png'],'-dpng','-r400');
        resFig.MenuBar = 'figure'; 
        resFig.ToolBar = 'figure';
        savefig(resFig,[basePath '\' subFolderUID ImageFileName '.dirDist.fig']);
        resFig.MenuBar = 'none'; 
        resFig.ToolBar = 'none';        
    end
    
    %the results are written to the results list array and saved to file
    resultsList = [{imageFileNameList{i}}, {CAS3d}, {CAS3dIntWeight}, {mainDirectionDeg},{mainElevationDeg},{weightedPeakLocs(1)},{peakLocs(1)},{saturation},{background},{meanSL}];
    %fprintf(resultsFileID,'%s;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f\n',resultsList{:});
    
    %fprintf(resultsFileID,'%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n',resultsList{:});
    fprintf(resultsFileID,resultsFileDataFormat,resultsList{:});
    waitbar(((i)/numel(imageFileNameList)),waitBar1,['Analyzing Images:' num2str(i) '/' num2str(numel(imageFileNameList))]);
end

%perform clean up
fclose(resultsFileID);
close(waitBar1);

%if run from the launcher, disable the "working state" circle gif
try
    src.Parent.Children(end-1).Visible = 'off';
catch
end

%display end message
disp('Analysis complete!');
if wCount>0
    disp(['With ' num2str(wCount) ' stacks with warnings']);
end

endMessage = ['Find your log file "' resultsFileName '" and figures at' newline basePath];

disp(endMessage);

messageButtonTexts = [{'Browse to folder'},{'OK'}];
messageBoxObject = questdlg(endMessage, 'Analysis complete', messageButtonTexts{1}, messageButtonTexts{2}, messageButtonTexts{1});

close(resFig);
if strcmp(messageBoxObject, messageButtonTexts{1})
    winopen(basePath);
end

function saveFFTasTiff(inputmatrixOrig)

%matlab can only export 16 bit tiffs...
inputmatrix = uint16(inputmatrixOrig);


tiff = inputmatrix(:,:,1);
imwrite(tiff,"FFTbufferFile.tiff",'WriteMode','overwrite');
for i=2:size(inputmatrix,3)
    tiff = inputmatrix(:,:,i);
    imwrite(tiff,"FFTbufferFile.tiff",'WriteMode','append');
end

end

function [xVector,yVector,zVector] = processIsotropicData(filtImagerffts)

intensityAtRadius = filtImagerffts(round(size(filtImagerffts,1)/2):end,round(size(filtImagerffts,2)/2),round(size(filtImagerffts,3)/2));
figure
plot(intensityAtRadius);
[peaks,locs] = findpeaks(intensityAtRadius);

xVector = locs(1);
yVector = 0;
zVector = 0;

end

function listSelection_callback(src, event)
global fileList;
global filteredFileList
filteredFileList = src.String(src.Value);
end

function adaptTableSize_callback(src,event)
global bottomOffset;
src.Children(end).Position = [0 bottomOffset src.Position(3) src.Position(4)-bottomOffset];
end

function filtButton_callback(src,event)
global fileList;
global filterString;
filterString = src.Parent.Children(end-1).String;
src.Parent.Children(end).String = fileList(find(contains(fileList, filterString)));
src.Parent.Children(end).Value = 1;
end

function clearFiltButton_callback(src,event)
global basePath;
global fileList;
src.Parent.Children(end).String = fileList;
src.Parent.Children(end-1).String = '';
end

function closebutton_callback(src,event)
close(src.Parent)
global bOK
bOK = 1;
end

function saveFigureCheckbox_callback(src,event)
global bSaveFigure;
bSaveFigure = src.Value
end

function systemSep = getSystemCSVseparator()
[status, cmdout]=system('REG QUERY "HKEY_CURRENT_USER\Control Panel\International" /v sList');
if status == 0
    %disp(cmdout);
    splitCmdOut = strsplit(cmdout);
    systemSep = splitCmdOut{end-1};
    %disp(systemSep);
else
    systemSep = ',';
    warning("System spreadsheet could not be detected, defaulting to ','");
end
end

function hiddenFileList = hideBasePath(fileList, basePath)


hiddenFileList = fileList;
for i=1:numel(fileList)
    hiddenFileList(i) = {fileList{i}(numel(basePath)+2:end)};
end

end

function reRangedDir = reRangeDir(angle)
if angle >= 180
    reRangedDir = angle-180;
elseif angle < 0
    reRangedDir = angle+180;
else
    reRangedDir = angle;
end
end

function reRangedElev = reRangeElev(angle)
% if angle > 180
%     reRangedElev = angle-180;
% elseif angle <= 0
%     reRangedElev = angle+180;
% else
%     reRangedElev = angle;
% end
reRangedElev = angle-90;
end

function filteredMatrix = removeBrightestVoxels(inMatrix, fracRetained)
nToRemove = floor(numel(inMatrix)*(1-fracRetained));
disp('Removing bright spots...');
for i=1:nToRemove
    [~,maxIndex] = max(inMatrix(:));
    [x,y,z] = ind2sub(size(inMatrix), maxIndex);
    inMatrix(x,y,z) = 0;
    %fprintf('.');
end
filteredMatrix = inMatrix;
end
