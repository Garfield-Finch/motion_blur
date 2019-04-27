function imgnm = gen_nm(num, key)
    % key == 0, input image name
    % key == 1, output interpolation image name
    % key == 2, .flo file name
    % key == 3, _ker.png
    if key == 0
        s = num2str(num);
        len = length(s);
        for i = 1:4-len
            s = ['0', s];
        end
        imgnm = ['frame_', s, '.png'];
    elseif key == 1
        s = num2str(num);
        len = length(s);
        for i = 1:4-len-1
            s = ['0', s];
        end
        imgnm = ['frame_1', s, '.png'];
    elseif key == 2
        s = num2str(num);
        len = length(s);
        for i = 1:4-len
            s = ['0', s];
        end
        imgnm = ['frame_', s, '.flo'];
    elseif key == 3
        s = num2str(num);
        len = length(s);
        for i = 1:4-len
            s = ['0', s];
        end
        imgnm = ['frame_', s, '_ker.png'];
    else
        disp('KEY ERROR !!!')
    end
end