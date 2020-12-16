%%
% Dynamic Fluid Surface Reconstruction using Deep Neural Network
% Authors: S Thapa, N Li, J Ye
% CVPR 2020
% contact: sthapa5@lsu.edu
%%
close all 
clear
%% Parameter Setting
Phase = 'train';
% Phase = 'val'; 
W = 128;
H = 128;
nWaveSeq = 9;
img_folder = ['RGB/' Phase '/'];

%% Generate Dataset
load(['Mapping_struct_' Phase]);

Mapping_type = '0';
Data_result_folder = [Phase '/Dataset/'];
Data_result_folder_depth = [Data_result_folder 'depth/'];
Data_result_folder_warp = [Data_result_folder 'warp/'];
Data_result_folder_RGB = [Data_result_folder 'RGB/'];
render_folder = [Data_result_folder 'Render_Matlab/'];

if ~exist(render_folder,'file')
    mkdir(render_folder);
end

if ~exist(Data_result_folder_depth,'file')
    mkdir(Data_result_folder_depth);
end

if ~exist(Data_result_folder_warp,'file')
    mkdir(Data_result_folder_warp);
end

if ~exist(Data_result_folder_RGB,'file')
    mkdir(Data_result_folder_RGB);
end

N = length(Mapping_struct);

for i = 1:N
    im_name = Mapping_struct(i).Image{1};
%     if exist([Data_result_folder 'RGB/' im_name],'file')
%         continue;
%     end
    im_name_ori = im_name(1:end-4);
    im_rgb = double(imread([img_folder im_name]))./255;
    im_rgb = imresize(im_rgb, [W,H]);
    ori_waveFolder = Mapping_struct(i).WaveFolder;
    depth_folder = [ori_waveFolder 'depth/'];
    wave_folder = [ori_waveFolder 'warp/'];
    start_loc = Mapping_struct(i).IndexRange_start;
    depth_list = dir([depth_folder '*.npy']);
    depth_str = depth_list(1).name;
    depth_alpha = '_5.npy';
    
    A = start_loc:(start_loc+nWaveSeq-1);
    seq_n = Mapping_struct(i).SeqNumber;
    
    % Load Wave and depth from original wavesequence folder
    wave_batch = arrayfun(@(x) readNPY([wave_folder  seq_n depth_alpha(1:2) '_' num2str(x) '.npy']),A,'Uni',0);
    depth_batch_cell = arrayfun(@(x) readNPY([depth_folder  seq_n depth_alpha(1:2) '_' num2str(x) '.npy']),A,'Uni',0);
    depth_batch_mat = cell2mat(depth_batch_cell);
    depth_batch = reshape(depth_batch_mat,[H,W,length(A)]);
    % Save Depth to datasest
    writeNPY(depth_batch,[Data_result_folder_depth im_name_ori '_' Mapping_type depth_alpha]);
    disp(['# Img ' num2str(i) ' : ' im_name]);   
    
    % Render refracted images for each background pattern
    for  j = A
        count = j - start_loc + 1;
        warp_xy = wave_batch{count};
        out_im = simulate(im_rgb, warp_xy(:,1:W),warp_xy(:,W+1:end), false,i);
        save_name = [im_name_ori '_' Mapping_type '_' num2str(count) depth_alpha(1:2)];
        imwrite(out_im, [render_folder save_name '.png']);
        writeNPY(warp_xy,[Data_result_folder 'warp/' save_name '_warp.npy']);        
%         pause(1/25);
%         figure(1),
%         subplot(1,2,1),imshow(im_rgb)
%         subplot(1,2,2),imshow(out_im)
    end
    imwrite(im_rgb,[Data_result_folder 'RGB/' im_name]);
end


