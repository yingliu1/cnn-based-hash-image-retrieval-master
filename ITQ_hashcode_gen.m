clear,clc

gallery_fusefeat_file = '../result/gallery.h5';
probe_fusefeat_file = '../result/probe.h5';
deep_feat = '/deep_f';

% 读取h5文件中的深度特征，由于python与matlab矩阵存储方式不同因而需要进行矩阵维度转换
h5disp(gallery_fusefeat_file);
gallery_data = double(permute(h5read(gallery_fusefeat_file, deep_feat), [2,1]));

h5disp(probe_fusefeat_file);
probe_data = double(permute(h5read(probe_fusefeat_file, deep_feat), [2,1]));

% 使用ITQ方法进行哈希编码，编码位数为48bit
disp('---------------------- generating hashcode ----------------------------')
bit = 48;
[gallery_code, probe_code] = compressITQ(bit, gallery_data, probe_data);
gallery_code = permute(int8(gallery_code), [2,1]);
probe_code = permute(int8(probe_code), [2,1]);

% 哈希码特征保存在h5文件中的字段名
hash_code = strcat('/hash', num2str(bit));

h5create(gallery_fusefeat_file, hash_code, size(gallery_code)); % 在h5文件中创建哈希码字段
h5write(gallery_fusefeat_file, hash_code, gallery_code); % 将哈希特征写入h5文件中
h5disp(gallery_fusefeat_file);

h5create(probe_fusefeat_file, hash_code, size(probe_code));
h5write(probe_fusefeat_file, hash_code, probe_code);
h5disp(probe_fusefeat_file);
disp('---------------------- hashcode has been saved! ----------------------')

