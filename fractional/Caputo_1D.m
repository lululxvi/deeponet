tic
space_paras = [10, 2];
space_type = 'Orthogonal';
training_set(10000, 15, 10, 10, '1D_Caputo', space_type,  space_paras);
test_set(10000, 15, 10, 10, '1D_Caputo',  space_type, space_paras);
toc
