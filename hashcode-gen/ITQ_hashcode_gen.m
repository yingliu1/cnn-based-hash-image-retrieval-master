clear,clc

gallery_fusefeat_file = '../result/gallery.h5';
probe_fusefeat_file = '../result/probe.h5';
deep_feat = '/deep_f';

% ��ȡh5�ļ��е��������������python��matlab����洢��ʽ��ͬ�����Ҫ���о���ά��ת��
h5disp(gallery_fusefeat_file);
gallery_data = double(permute(h5read(gallery_fusefeat_file, deep_feat), [2,1]));

h5disp(probe_fusefeat_file);
probe_data = double(permute(h5read(probe_fusefeat_file, deep_feat), [2,1]));

% ʹ��ITQ�������й�ϣ���룬����λ��Ϊ48bit
disp('---------------------- generating hashcode ----------------------------')
bit = 48;
[gallery_code, probe_code] = compressITQ(bit, gallery_data, probe_data);
gallery_code = permute(int8(gallery_code), [2,1]);
probe_code = permute(int8(probe_code), [2,1]);

% ��ϣ������������h5�ļ��е��ֶ���
hash_code = strcat('/hash', num2str(bit));

h5create(gallery_fusefeat_file, hash_code, size(gallery_code)); % ��h5�ļ��д�����ϣ���ֶ�
h5write(gallery_fusefeat_file, hash_code, gallery_code); % ����ϣ����д��h5�ļ���
h5disp(gallery_fusefeat_file);

h5create(probe_fusefeat_file, hash_code, size(probe_code));
h5write(probe_fusefeat_file, hash_code, probe_code);
h5disp(probe_fusefeat_file);
disp('---------------------- hashcode has been saved! ----------------------')

