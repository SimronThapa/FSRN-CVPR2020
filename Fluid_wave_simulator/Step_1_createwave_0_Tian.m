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
wave_folder = [Phase '/WaveSequences/Wave_Tian/Seq_' seq_Num '/'];
depth_folder = [wave_folder 'depth/'];
warp_folder = [wave_folder 'warp/'];
% Set Image size (W,H) and wave sequence number(Nframe)for this type
W = 128;
H = W;
nFrame = 10;

%% Generate Waves
if ~exist(depth_folder,'file')
    mkdir(depth_folder);
end

if ~exist(warp_folder,'file')
    mkdir(warp_folder);
end

% img = rgb2gray(im2double(imread('binary.jpg')));
% img = img(320:463, :);
rgb = im2double(imread('tex1.jpg'));
rgb = imresize(rgb, [W,H]);
img = rgb2gray(rgb);

%% Attributes
%alpha = 5(0.1563);
randn('state', 1);
c = 0.8;
alpha = 5;
level_factor = 0.15;

%%
[warp, levels] = getWarpWater(img, nFrame + 1, c, alpha);
% frames = simulate(rgb, warp, true);

[X,Y] = meshgrid(1:W);
% [X, Y] = meshgrid(linspace(-1,1,256),linspace(-1,1,256));
h = figure(1);
hh = subplot(2,3,1);handle = surf (X,Y,levels(:,:,1),'FaceColor','interp','edgecolor','none','edgelighting','none');
colormap(cold);
%     camlight('headlight');
lighting('gouraud');
shading('interp');

n1 = 1;
n2 = 1.33;
levels = levels*level_factor;
levels = levels(:,:,2:end);

% filename_depth=[depth_folder 'Seq_' seq_Num  '_' num2str(alpha) '.npy'];
% writeNPY(levels,filename_depth);

for i = 1: nFrame
    disp(num2str(i));
    zh = levels(:,:,i);
    zh_new = zh;
    zh_nl = zh_new+alpha*4;
%     zh_range = max(zh(:)) -  min(zh(:));

    [warp_map] = raytracing_im_generator_ST(rgb,zh_nl,n1,n2,X,Y);
    filename_warp=[warp_folder 'Seq_' seq_Num '_' num2str(alpha) '_' num2str(i) '.npy'];
    writeNPY(warp_map,filename_warp);
    % Save Depth array
    filename_depth=[depth_folder 'Seq_' seq_Num  '_' num2str(alpha) '_' num2str(i) '.npy'];
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



