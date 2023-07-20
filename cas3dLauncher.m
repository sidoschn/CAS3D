
disp('CAS3d Tool v4.43');
disp('Institute of Medical Biotechnology, FAU Erlangen');
disp('loading GUI...');

global basePath;
global bEvalRunning;

% abort functionality was scrapped in 4.43

%global bAbortEval;
%bAbortEval = 0;

launcherfigure = uifigure;
launcherfigure.Color = [1 1 1];
launcherfigure.Position = [256 256 512 512];
launcherfigure.Name = 'CAS3d Tool v4.43';
launcherfigure.NumberTitle = 'off';
launcherfigure.ToolBar = 'none';
launcherfigure.MenuBar = 'none';
launcherfigure.Resize = 'off';

%bgImage = imshow("splash.png", "Border", "tight");
bgImage = uiimage(launcherfigure);
bgImage.ImageSource = "splash.png";
bgImage.Position = [0 0 512 512];


waitImage = uiimage(launcherfigure, 'Visible', 'off');
waitImage.ImageSource = "BrokenCircle.gif";
waitImage.Position = [50 64 64 64];


bRun = uibutton(launcherfigure, 'ButtonPushedFcn',@runButton_callback);
%bRun = uicontrol(launcherfigure,'Style','pushbutton','Callback',@runButton_callback);
bRun.Text = "Run evaluation";
bRun.Position = [20 20 120 20];

%bDocu = uicontrol(launcherfigure,'Style','pushbutton','Callback',@docuButton_callback);
bDocu = uibutton(launcherfigure, 'ButtonPushedFcn',@docuButton_callback);
bDocu.Text = "Open documentation";
bDocu.Position = [180 20 120 20];

% abort functionality was scrapped in 4.43

% bAbort = uibutton(launcherfigure, 'ButtonPushedFcn',@abortButton_callback);
% bAbort.Text = "X";
% bAbort.Position = [120 80 32 32];
% bAbort.BackgroundColor = [1 1 1];
% bAbort.FontWeight = "bold";
% bAbort.FontSize = 18;




function runButton_callback(src,event)
global basePath;
global bEvalRunning;

src.Enable = 'off';
bEvalRunning = 1;
cas3dTool4;
bEvalRunning = 0;
src.Enable = 'on';

end

function abortButton_callback(src,event)
global bAbortEval;
global bEvalRunning;

src.Enable = 'off';
bAbortEval = 1;

while bEvalRunning==1
    
end

src.Parent.Children(end-1).Visible = 'off';
src.Enable = 'on';
end

function docuButton_callback(src,event)
winopen('Documentation.pdf');
disp("opening documentation...");
end