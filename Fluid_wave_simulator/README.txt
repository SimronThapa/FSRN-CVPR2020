Running steps:

Create a 'RGB' folder with 'train' and 'val' sub-folders
Add background pattern to 'RGB' folder

Generate Training set:
	- Set variable 'Phase' to 'train' in all the matlab file
Generate Validation set:
	- Set variable 'Phase' to 'val'	in all the matlab file

0: Add npy-matlab-master to the matlab path

1: Generate waves for each wave type
	- Ocean wave: run Step_1_createwave_0_Ocean
	- Ripple wave: run Step_1_createwave_0_Ripple
	- Waves from "Seeing through Water:xxx" paper: run Step_1_createwave_0_Tian
	
	Will save waves in the 'train\WaveSequences\' or 'val\WaveSequences\' folder
	
2: Generate Mapping matrix for the rendering steps:
	- Run Step_2_Image_Wave_Index_Mapping
	
	Will save matrix to the 'Mapping_struct_train.mat' or 'Mapping_struct_val.mat' 
	
3. Rendering refracted images for each pattern:
	- Run Step_3_Render_dataset
	
	Will save waves in the 'train\Dataset\' or 'val\Dataset\' folder
