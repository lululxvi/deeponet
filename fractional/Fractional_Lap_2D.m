tic
space_paras = [14, 2];
space_type = 'Orthogonal';
training_set(5000, 15, 15, 10, '2D_fLap_disk', space_type,  space_paras);
test_set(5000, 15, 15, 10, '2D_fLap_disk',  space_type, space_paras);
toc
