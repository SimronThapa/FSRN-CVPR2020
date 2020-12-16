%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
close all
clear

%% Parameter Setting
seq_Num = '9';
Phase = 'train'; 
% Phase = 'val'; %uncomment this for generating validation set
% wave_folder = [Phase '/WaveSequences/Wave_Ripple/Seq_' seq_Num '/'];
% depth_folder = [wave_folder 'depth/'];
% warp_folder = [wave_folder 'warp/'];
depth_folder = fullfile(Phase,'WaveSequences', 'Wave_Ripple',strcat('Seq_',seq_Num),'depth');
warp_folder = fullfile(Phase,'WaveSequences', 'Wave_Ripple',strcat('Seq_',seq_Num),'warp');
% Set Image size (W,H) and wave sequence number (nFrame) for this type
W = 128;
H = W;
nFrame = 100;

%% Generate Waves
if ~exist(depth_folder,'file')
    mkdir(depth_folder);
end

if ~exist(warp_folder,'file')
    mkdir(warp_folder);
end

xx=-(round(W*0.5)-1):1:round(W*0.5);
yy=xx;
[XX,YY] = meshgrid(xx,yy); % create rectangular mesh



%% Attributes
% Seq_9 
alpha = 5;
c = 0.3;
level_factor = 0.25;
R=sqrt((XX+100).^2+(YY+150).^2)/8; %radius %very new 4300-4499
R1=sqrt((XX-100).^2+(YY+150).^2)/8;
k=0.08; % wave vector
R2=sqrt((XX-100).^2+(YY-20).^2)/8;
k2=0.1;

%% Other Attributes
rgb = im2double(imread('tex1.jpg'));
pattern_im = imresize(rgb, [W,H]);
rgb = pattern_im;
img = rgb2gray(rgb);

n1 = 1;
n2 = 1.33;
D0 = 4*alpha;

%% Wave Generation
[warp, levels] = getTwoRippleWave(img, nFrame + 1, alpha, c, R, R1, k);
% [warp, levels] = getThreeRippleWave(img, nFrame + 1, alpha, c, R, R1, k, k2,R2);

levels = levels*level_factor;
levels = levels(:,:,2:end);

[X,Y] = meshgrid(1:W);
h = figure(1);
hh = subplot(2,3,1);handle = surf (X,Y,levels(:,:,1),'FaceColor','interp','edgecolor','none','edgelighting','none');
colormap(cold);
lighting('gouraud');
shading('interp');

% filename_depth=[depth_folder 'Seq_' seq_Num  '_' num2str(alpha) '.npy'];
% writeNPY(levels,filename_depth);

for i = 1: nFrame
    disp(num2str(i));
    zh = levels(:,:,i);
    zh_new = zh;
    zh_nl = zh_new+D0;    
    
    [warp_map] = raytracing_im_generator_ST(rgb,zh_nl,n1,n2,X,Y);    
    %filename_warp=[warp_folder 'Seq_' seq_Num '_' num2str(alpha) '_' num2str(i) '.npy'];
    filename_warp=fullfile(warp_folder,strcat('Seq_', seq_Num, '_', num2str(alpha), '_', num2str(i) ,'.npy'));
    
    writeNPY(warp_map,filename_warp);
    % Save Depth array
    %filename_depth=[depth_folder 'Seq_' seq_Num  '_' num2str(alpha) '_' num2str(i) '.npy'];
    filename_depth=fullfile(depth_folder, strcat('Seq_', seq_Num,  '_', num2str(alpha), '_', num2str(i), '.npy'));
    writeNPY(zh,filename_depth);       
       
    zh_range = max(zh(:)) -  min(zh(:));
    set(handle,'zdata',zh_new); 
    title(hh,sprintf('# %d: [%.3f , %.3f] --- %.3f', i, min(zh(:)),max(zh(:)), zh_range)); 
    set(gca,'zlim',[min(levels(:)),max(levels(:))]);
    warp_x = warp.Xs(:, :, i)*level_factor;
    warp_y = warp.Ys(:, :, i)*level_factor;
    frames = simulate(rgb, warp_x,warp_y, false,i);
    recovered_in = simulate(frames, -warp_x,-warp_y, false,i);
    
    subplot(2,3,2),imshow(frames);title('Distorted by Raytracing') 
    subplot(2,3,3),imshow(frames),title('Distorted by Warp')    
    subplot(2,3,4),quiver(warp_x,warp_y),axis equal%plot(warp_map,'DecimationFactor',[10 10],'ScaleFactor',10)
    subplot(2,3,5),imshow(recovered_in),title('Recoverd By Warp') 
    subplot(2,3,6),imshow(rgb),title('Undistored GT')  
    pause(1/25);
end


  

