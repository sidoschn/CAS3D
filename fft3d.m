

testGrid2 = bfopen('D:\ImageDump01\Goossens\done\193\181122_193 fiber 1-3_14-37-28\14-37-28_193 fiber 1-3_PMT - PMT [PMT1] _C00_xyz Stage Z0000.ome.tif');
imageMetaData = testGrid2{1,4};
stackChannelCount = imageMetaData.getPixelsSizeC(0).getValue();
    stackSliceCount = imageMetaData.getPixelsSizeZ(0).getValue();
    stackWidth = imageMetaData.getPixelsSizeX(0).getValue();
    stackHeight = imageMetaData.getPixelsSizeY(0).getValue();
testGrid2pixelsRaw = zeros(stackHeight,stackWidth,stackSliceCount);

for j=1:32
        testGrid2pixelsRaw(:,:,j)=testGrid2{1,1}{(j*3),1};
end
    
fft = fftn(testGrid2pixelsRaw);


ffts = fftshift(fft,2);
ffts = fftshift(ffts,1);
ffts = fftshift(ffts,3);

iffts = imag(ffts);
rffts = real(ffts);
implay(log(abs(rffts)));
histogram(log(abs(rffts)))