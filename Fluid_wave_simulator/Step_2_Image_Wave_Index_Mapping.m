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
img_folder = ['RGB/' Phase '/'];


wave_folder = [Phase '/WaveSequences/'];
% Set number of waves per image(background pattern)
nWaveSeq = 10;
% Set number of images in Training or Val set
nPattern = 4359;   %val = 998, train = 4359

%% Generate The Mapping matrix
img_list = dir([img_folder '*.jpg']);
img_namelist = {img_list.name};
img_namelist = img_namelist';
wavetype_list = dir(wave_folder);
% For Mac remove hidden files if any ex: starting with .
wavetype_list(strncmp({wavetype_list.name}, '.', 1)) = [];
wavetype_list = wavetype_list(3:end);

Mapping_struct= [];
imNum = length(img_namelist);
nWaveType = length(wavetype_list);

if imNum < nPattern
    disp('Do not have enough images. Will use all availab images in the folder.');
    nPattern = imNum;
end

for i = 1:nPattern
    
    this_waveType = wavetype_list(randperm(nWaveType,1));
    this_AllSeq = dir([wave_folder this_waveType.name]);
    this_AllSeq = this_AllSeq(3:end);

    
    this_Seq = this_AllSeq(randperm(length(this_AllSeq),1));
    this_SeqFolder = [this_Seq.folder '/' this_Seq.name '/'];
    this_WaveList = dir([this_SeqFolder 'warp/*.npy']);
    this_nWave = length(this_WaveList);
    this_nvalidIdx = this_nWave - nWaveSeq + 1;
    if this_nvalidIdx < 1
        error(['nWaveSeq must be less or equal to ' num2str(this_nWave)]);
    end
    this_startIdx = randperm(this_nvalidIdx,1);
    Mapping_struct(i).Image =  img_namelist(i);
    Mapping_struct(i).WaveFolder = this_SeqFolder;
    Mapping_struct(i).WaveType = this_waveType.name;
    Mapping_struct(i).SeqNumber = this_Seq.name;
    Mapping_struct(i).IndexRange_start =  this_startIdx;    
end
save( ['Mapping_struct_' Phase], 'Mapping_struct');